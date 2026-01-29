"""
Simple script to run a notebook multiple times.
Usage: python run_notebook.py <notebook_name> <num_runs>
Example: python run_notebook.py imperative.ipynb 10
"""
import sys
import subprocess
from pathlib import Path

def run_notebook(notebook_path: str, run_number: int) -> bool:
    """Execute a notebook once using nbclient."""
    print(f"\n{'='*70}")
    print(f"Run {run_number}: {notebook_path}")
    print(f"{'='*70}")
    
    try:
        import nbformat
        from nbclient import NotebookClient
        
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Execute the notebook
        client = NotebookClient(nb, timeout=7200)  # 2 hour timeout
        client.execute()
        
        # Save the executed notebook
        with open(notebook_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        
        print(f"✅ Completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python run_notebook.py <notebook_name> <num_runs>")
        print("Example: python run_notebook.py imperative.ipynb 10")
        sys.exit(1)
    
    notebook_path = sys.argv[1]
    num_runs = int(sys.argv[2])
    
    if not Path(notebook_path).exists():
        print(f"❌ Error: Notebook '{notebook_path}' not found")
        sys.exit(1)
    
    print(f"Starting {num_runs} runs of {notebook_path}")
    
    successful = 0
    failed = 0
    
    for i in range(num_runs):
        if run_notebook(notebook_path, i + 1):
            successful += 1
        else:
            failed += 1
    
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    print(f"Total runs: {num_runs}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    
    # Show results file location
    notebook_name = Path(notebook_path).stem
    results_file = f"results_{notebook_name}.csv"
    if Path(results_file).exists():
        print(f"\n✅ Results saved to: {results_file}")

if __name__ == "__main__":
    main()
