# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""LiteRT LM Evaluation Runner.

This script acts as a unified entry point for evaluating LiteRT models using
various evaluation frameworks (like `lm_eval`). It provides a consistent set of
command-line flags across different underlying frameworks, simplifying the
pipeline setup.

It parses the unified flags, maps them to the specific arguments required by
the chosen framework, and then delegates execution to that framework's CLI.

Usage example:
  bazel run //python/litert_lm_eval:litert_lm_eval \
      --model_path /path/to/model.litertlm \
      --tasks mmlu,gsm8k \
      --backend CPU \
      --output_path /path/to/save/results.json

  # Using the escape hatch for framework-specific arguments
  bazel run //python/litert_lm_eval:litert_lm_eval \
      --model_path /path/to/model.litertlm \
      --tasks mmlu \
      --framework_args "limit=10"
"""

import argparse
import json
import sys
from typing import Any

import lm_eval
import lm_eval.tasks

from litert_lm_eval import utils
from litert_lm_eval.runners.lm_eval_runner import litert_lm_model  # pylint: disable=unused-import

# Keys used to merge the evaluation results from separate scoring and sampling
# runs.
_EVAL_RESULT_KEYS = (
    "results",
    "groups",
    "group_subtasks",
    "configs",
    "versions",
    "n-shot",
    "higher_is_better",
    "n-samples",
    "samples",
)


def main():
  parser = argparse.ArgumentParser(description="LiteRT LM Eval Runner")

  parser.add_argument(
      "--model_path", type=str, required=True, help="Path to the model file."
  )
  parser.add_argument(
      "--tasks",
      type=str,
      required=True,
      help="Comma-separated list of tasks to run (e.g., 'mmlu,gsm8k').",
  )
  parser.add_argument(
      "--backend",
      type=str,
      default="CPU",
      choices=["CPU", "GPU"],
      help="Backend to use (e.g., 'CPU', 'GPU').",
  )

  parser.add_argument(
      "--num_fewshot",
      type=int,
      default=None,
      help="Number of examples in few-shot context.",
  )
  parser.add_argument(
      "--limit",
      type=float,
      default=None,
      help="Limit examples per task (integer count or fraction).",
  )
  parser.add_argument(
      "--tokenizer",
      type=str,
      default=None,
      help="Optional path or name of the tokenizer to use.",
  )

  # Escape hatch
  parser.add_argument(
      "--framework_args",
      type=str,
      default="",
      help=(
          "Additional arguments to pass strictly to the model constructor "
          "(comma-separated key=value pairs or flags)."
      ),
  )

  parser.add_argument(
      "--output_path",
      type=str,
      default=None,
      help="Path to save the evaluation results as a JSON file.",
  )
  parser.add_argument(
      "--apply_chat_template",
      type=lambda x: str(x).lower() in ("true", "1", "yes"),
      default=False,
      help=(
          "Specifies whether to apply a chat template to the prompt. Note: This"
          " is only useful for scoring tasks, and requires the --tokenizer flag"
          " to be provided. For sampling/generation tasks, a chat template is"
          " intrinsically applied by default via the LiteRT LM runner."
      ),
  )

  args, unknown = parser.parse_known_args()

  def _is_generate_until_task(task_dict: dict[str, Any]) -> bool:
    """Recursively checks if any task in task_dict is a 'generate_until' task.

    Args:
      task_dict: A nested dictionary of task objects.

    Returns:
      True if the output type is 'generate_until', False otherwise.
    """
    for _, task_obj in task_dict.items():
      if isinstance(task_obj, dict):
        if _is_generate_until_task(task_obj):
          return True
      else:
        if task_obj.get_config("output_type") == "generate_until":
          return True
    return False

  def _merge_results(base_results, new_results):
    """Merges evaluation results from two separate runs.

    This is necessary because scoring tasks and sampling tasks are evaluated in
    separate `lm_eval.simple_evaluate` calls and must be combined.

    Args:
      base_results: The initial results dictionary (can be None).
      new_results: The results dictionary to merge into base_results.

    Returns:
      A unified dictionary containing the merged results.
    """
    if not base_results:
      return new_results
    if not new_results:
      return base_results

    merged = base_results.copy()
    for key in _EVAL_RESULT_KEYS:
      if key in new_results:
        if key not in merged:
          merged[key] = {}
        merged[key].update(new_results[key])
    return merged

  # Construct the model_args string required by lm_eval.
  model_args_str = f"model_path={args.model_path},backend={args.backend}"

  if args.tokenizer:
    model_args_str += f",tokenizer={args.tokenizer}"

  if args.framework_args:
    model_args_str += f",{args.framework_args}"

  tasks = args.tasks.split(",") if args.tasks else []

  # Parse unknown args into kwargs for simple_evaluate.
  kwargs = utils.parse_unknown_args(unknown)

  task_manager = lm_eval.tasks.TaskManager()
  scoring_tasks = []
  sampling_tasks = []

  for task_name in tasks:
    t_dict = lm_eval.tasks.get_task_dict([task_name], task_manager)
    if _is_generate_until_task(t_dict):
      sampling_tasks.append(task_name)
    else:
      scoring_tasks.append(task_name)

  results = None

  if scoring_tasks:
    print(
        "Running evaluation with model 'litert_lm' on scoring tasks:"
        f" {scoring_tasks}"
    )
    scoring_results = lm_eval.simple_evaluate(
        model="litert_lm",
        model_args=model_args_str,
        tasks=scoring_tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        apply_chat_template=args.apply_chat_template,
        **kwargs,  # Pass any remaining flags.
    )
    results = _merge_results(results, scoring_results)

  if sampling_tasks:
    print(
        "Running evaluation with model 'litert_lm' on sampling tasks:"
        f" {sampling_tasks}"
    )
    # Force apply_chat_template=False for sampling tasks because the litert_lm
    # model runner already applies chat templates internally via the
    # Conversation API. Enabling it on the pipeline might result in
    # double-templating.
    sampling_kwargs = kwargs.copy()
    sampling_kwargs.pop("apply_chat_template", None)

    sampling_results = lm_eval.simple_evaluate(
        model="litert_lm",
        model_args=model_args_str,
        tasks=sampling_tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        apply_chat_template=False,
        **sampling_kwargs,  # Pass any remaining flags.
    )
    results = _merge_results(results, sampling_results)

  if results is not None:
    print(json.dumps(results["results"], indent=2, default=str))
    if args.output_path:
      with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
      print(f"\nResults successfully saved to {args.output_path}")
    print("\nEvaluation successful.")


if __name__ == "__main__":
  main()
