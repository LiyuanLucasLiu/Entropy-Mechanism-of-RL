# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
from collections import defaultdict
from verl import DataProto
from verl.utils.reward_score import _default_compute_score


class DAPORewardManager:
    """The reward manager.
    """

    def __init__(self,
                 tokenizer,
                 num_examine,
                 compute_score=None,
                 reward_fn_key='data_source',
                 max_resp_len=None,
                 overlong_buffer_cfg=None,
                 timeout=50,
                 num_processes=64):
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.max_resp_len = max_resp_len
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.timeout = timeout
        self.num_processes = num_processes

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"

    def single_compute_score(self, data_source, response_str, ground_truth, extra_info):
        try:
            return self.compute_score(data_source, response_str, ground_truth, extra_info)
        except Exception as e:
            print("Error in compute_score_fn:", e)
            return None
    
    
    # def parallel_compute_sync(self, data_sources, respnses_str, ground_truths, extra_infos):
    #     with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
    #         futures = []
    #         for data_source, response_str, ground_truth, extra_info in zip(data_sources, respnses_str, ground_truths, extra_infos):
    #             futures.append(executor.submit(self.single_compute_score, data_source, response_str, ground_truth, extra_info))

    #         results = []
    #         for future in tqdm(futures, desc="Reward Scoring"):
    #             try:
    #                 results.append(future.result(timeout=self.timeout))
    #             except TimeoutError:
    #                 results.append(None)
    #             except Exception as e:
    #                 print("Error in compute score:", e)
    #                 results.append(None)
    
    def parallel_compute_sync(self, data_sources, responses_str, ground_truths, extra_infos):
        results = [None] * len(data_sources)
        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            future_to_index = {
                executor.submit(self.single_compute_score, ds, rs, gt, ei): i
                for i, (ds, rs, gt, ei) in enumerate(zip(data_sources, responses_str, ground_truths, extra_infos))
            }

            for future in tqdm(as_completed(future_to_index), total=len(future_to_index), desc="Reward Scoring"):
                idx = future_to_index[future]
                try:
                    results[idx] = future.result(timeout=self.timeout)
                except TimeoutError:
                    print(f"Timeout for item {idx}")
                    results[idx] = None
                except Exception as e:
                    print(f"Exception at {idx}: {e}")
                    results[idx] = None
        return results

    def __call__(self, data: DataProto, return_dict: bool = False):
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}

        # batched scoring
        response_ids = data.batch['responses']
        responses_str = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truths = [data_item.non_tensor_batch['reward_model']['ground_truth'] for data_item in data]
        data_sources = data.non_tensor_batch['data_source']
        extra_infos = data.non_tensor_batch.get('extra_info', None)

        assert len(responses_str) == len(ground_truths) == len(data_sources) == len(extra_infos)
        
        results = self.parallel_compute_sync(data_sources, responses_str, ground_truths, extra_infos)
        
        for i, data_item in enumerate(data):
            result = results[i]

            prompt_ids = data_item.batch['prompts']
            response_ids = data_item.batch['responses']
            attention_mask = data_item.batch['attention_mask']

            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = attention_mask[:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            valid_response_length = attention_mask[prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            eos_token = self.tokenizer.eos_token
            if response_str.endswith(eos_token):
                response_str = response_str[:-len(eos_token)]

            score = result.get("score", 0.0) if isinstance(result, dict) else 0.0
            if isinstance(result, dict):
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                reward_extra_info["score"].append(score)
                reward_extra_info["acc"].append(False)
                reward_extra_info["extracted_gt"].append(ground_truths[i])
                reward_extra_info["extracted_pred"].append(None)

            reward = score
            if self.overlong_buffer_cfg.enable:
                overlong_buffer_len = self.overlong_buffer_cfg.len
                expected_len = self.max_resp_len - overlong_buffer_len
                exceed_len = valid_response_length - expected_len
                overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
                overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
                reward += overlong_reward
                if self.overlong_buffer_cfg.log:
                    reward_extra_info["overlong_reward"].append(overlong_reward)
                    reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            data_source = data_sources[i]
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truths[i])
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print(f"[score]", score)

        # save
        with open("./outputs/reward_score.jsonl", "a") as f:
            for i in range(len(responses_str)):
                json.dump({
                    "question": extra_infos[i]['question'],
                    "pred": responses_str[i],
                    "gold": ground_truths[i],
                    "extracted_gt": reward_extra_info["extracted_gt"][i],
                    "extracted_pred": reward_extra_info["extracted_pred"][i],
                    "acc": reward_extra_info["acc"][i]
                }, f, ensure_ascii=False)
                f.write("\n")

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
