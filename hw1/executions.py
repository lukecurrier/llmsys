import os
import json
import subprocess

def execute_tests():
    total_tests = 0
    total_passed = 0
    if not os.path.exists("test_results"):
            os.makedirs("test_results")
    for completion_file in os.listdir("completions"):
        with open(f"completions/{completion_file}", 'r') as f:
                completion_data = json.load(f)
        curr_tests = completion_data['Tests']
        solution_output = []
        for idx, ind_completion in enumerate(completion_data['Completions']):
            with open("temp_test.py", 'w') as f:
                f.write(ind_completion)
                f.write('\n\n')
                f.write(curr_tests)
            try:
                result = subprocess.run(
                    ['python3', "temp_test.py"],
                    capture_output=True, text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    total_passed = total_passed + 1
                    total_tests = total_tests + 1
                    result_data = {
                        'solution_name': f"test_{idx}",
                        'pass': True,
                        'output': result.stdout
                    }
                else:
                    total_tests = total_tests + 1
                    result_data = {
                        'solution_name': f"test_{idx}",
                        'pass': False,
                        'output': result.stderr
                    }
                solution_output.append(result_data)
            except subprocess.TimeoutExpired:
                return False, "Timeout expired after 5 seconds."
        with open(f"test_results/{completion_file}", 'w') as result_file:
                json.dump(solution_output, result_file, indent=4)
    return f"{total_passed/total_tests * 100}% pass rate"

print(execute_tests())
