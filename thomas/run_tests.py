"""
Runs tests for agent.py. Usage:

```bash
python3 run_tests.py \
    --model MODEL_NAME \
    --concurrency N \
    --num-samples M \
    --tests TESTS_FILE \
    --log LOG_FILE \
```

MODEL_NAME is the name of the model to use (and passed to agent.py).
N is the number of tests to run concurrently.
M is the number of attempts to make for each test case.
LOG_FILE is the path to a file where all interactions are logged.
TESTS_FILE is the path to a YAML file with the following format:

```yaml
- user_messages:
    - What are the flights from Boston to Dallas on January 1?
    - Book the first one, thanks.
  # Booking won't work -- the flight is full.
  expected_result: []
- user_messages:
    - What are the flights from Boston to Dallas on January 1?
    - Book the second one, thanks.
  expected_result: [2]
```

Thus each test case in the YAML file is a list of user messages to send to
the agent and the list of flight IDs that should be booked.

To run each test, the script starts agent.py as a subprocess, sends it the
user messages as input and then checks that the last line of output from
the agent is the list of flight IDs.

We log all interactions to LOG_FILE
"""

import asyncio
import yaml
import argparse
from pathlib import Path

from typing import List, Tuple
import re
import tqdm


async def run_test(
    model_name: str,
    user_messages: List[str],
    expected_result: List[int],
) -> Tuple[bool, str]:
    # Start the agent process
    process = await asyncio.create_subprocess_exec(
        "python3",
        "thomas.py",
        model_name,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    try:
        stdout, stderr = await process.communicate(
            "\n".join(user_messages).encode("utf-8")
        )
        stdout = stdout.decode("utf-8")
        stderr = stderr.decode("utf-8")

        # Extract the last line which should contain booked flights
        # Extract list of flight IDs using regex
        match = re.search(r"\[([\d,\s]*)\]$", stdout.strip())
        if match:
            booked_flights = [
                int(flight_id)
                for flight_id in match.group(1).split(",")
                if len(flight_id) > 0
            ]
        else:
            booked_flights = []

        success = booked_flights == expected_result

        log = f"Messages: {user_messages}\nExpected: {expected_result}\nGot: {booked_flights}\n"
        log += f"Stdout: {stdout}\n"
        if stderr:
            log += f"Stderr: {stderr}\n"
        return success, log

    except Exception as e:
        return False, f"Error running test: {str(e)}\n"

    finally:
        try:
            process.terminate()
            await process.wait()
        except:
            pass


async def main_with_args(
    model_name: str,
    tests_file: Path,
    log_file: Path,
    concurrency: int,
    num_samples: int,
):
    # Load test cases from YAML file
    with tests_file.open("r") as f:
        test_cases = yaml.safe_load(f)

    # Create semaphore to limit concurrent tests
    sem = asyncio.Semaphore(concurrency)

    # Run tests concurrently with semaphore
    async def run_test_with_sem(test_case, index):
        async with sem:
            success, log = await run_test(
                model_name, test_case["user_messages"], test_case["expected_result"]
            )
            return success, log, index

    # Create all test tasks
    tasks = [
        run_test_with_sem(test_case, i)
        for i, test_case in enumerate(test_cases * num_samples)
    ]

    # Run all tests with progress bar
    with tqdm.tqdm(total=len(tasks), desc="Running tests") as pbar, log_file.open(
        "w"
    ) as log_f:
        results = []
        for coro in asyncio.as_completed(tasks):
            success, log, index = await coro
            results.append((success, log))
            log_f.write(f"\nTest {index + 1}\n")
            log_f.write(f"Success: {success}\n")
            log_f.write(log)
            log_f.write("-" * 80 + "\n")
            log_f.flush()
            pbar.update(1)

    # Print summary
    num_passed = sum(1 for success, _ in results if success)
    print(f"Passed {num_passed} out of {len(results)} tests")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="llama3p1-8b-instruct",
        help="Name of the model to test",
    )
    parser.add_argument(
        "--tests", type=Path, default="tests.yaml", help="YAML file containing test cases"
    )
    parser.add_argument(
        "--log", type=Path, default="test_log.txt", help="File to write test results to"
    )
    parser.add_argument(
        "--concurrency", type=int, default=4, help="Number of tests to run concurrently"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of attempts to make for each test case",
    )
    args = parser.parse_args()

    asyncio.run(
        main_with_args(
            args.model, args.tests, args.log, args.concurrency, args.num_samples
        )
    )


if __name__ == "__main__":
    main()
