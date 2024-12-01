import io
import json
import sys
import multiprocessing


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def execute_code(code, output_queue):
    """Executes the given code and puts the result in the queue."""
    stdout_capture = io.StringIO()
    original_stdout = sys.stdout  # Save original stdout
    try:
        sys.stdout = stdout_capture  # Redirect stdout
        exec(code)  # Execute the code
        output_queue.put((True, stdout_capture.getvalue()))
    except Exception as e:
        output_queue.put((False, str(e)))
    finally:
        sys.stdout = original_stdout  # Always restore stdout

def get_python_code_output(code, timeout=3):
    """
    Runs Python code and captures output, skipping code that waits for stdin.
    Times out after the specified duration.
    """
    output_queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=execute_code, args=(code, output_queue))
    
    process.start()
    process.join(timeout)

    # If process is still alive after timeout, terminate it
    if process.is_alive():
        process.terminate()  # Kill the process
        process.join()  # Ensure cleanup
        return None  # Skipping code due to timeout

    # Retrieve results from the queue
    if not output_queue.empty():
        success, result = output_queue.get()
        return result if success else None

    return None
    
def extract_code_block(text, start_delimiter, end_delimiter):
    """Extracts text between start_delimiter and end_delimiter."""
    try:
        start_index = text.index(start_delimiter) + len(start_delimiter)
        end_index = text.index(end_delimiter, start_index)
        return text[start_index:end_index].strip()
    except ValueError:
        return None

def safe_parse_float(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None