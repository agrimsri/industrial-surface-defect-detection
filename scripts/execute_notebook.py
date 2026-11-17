import sys
import io
import base64
import contextlib
import nbformat as nbf
from IPython.core.interactiveshell import InteractiveShell
import matplotlib.pyplot as plt

def execute_and_save_notebook(notebook_path, output_path=None):
    if output_path is None:
        output_path = notebook_path
        
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbf.read(f, as_version=4)
        
    shell = InteractiveShell.instance()
    
    print(f"Executing {notebook_path} ({len(nb.cells)} cells)...")
    exec_count = 1
    
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type != 'code':
            continue
            
        code = cell.source
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        
        plt.close('all')
        
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            result = shell.run_cell(code)
            
        outputs = []
        stdout_text = stdout_buf.getvalue()
        stderr_text = stderr_buf.getvalue()
        
        if stdout_text:
            outputs.append(nbf.v4.new_output(output_type='stream', name='stdout', text=stdout_text))
        if stderr_text:
            outputs.append(nbf.v4.new_output(output_type='stream', name='stderr', text=stderr_text))
            
        # Check if matplotlib generated any figures
        fig_nums = plt.get_fignums()
        for fnum in fig_nums:
            fig = plt.figure(fnum)
            img_buf = io.BytesIO()
            fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=100)
            img_buf.seek(0)
            img_b64 = base64.b64encode(img_buf.read()).decode('utf-8')
            outputs.append(nbf.v4.new_output(
                output_type='display_data',
                data={'image/png': img_b64, 'text/plain': '<Figure size ...>'},
                metadata={}
            ))
        plt.close('all')
        
        if not result.success:
            print(f"Error in cell {idx}:\n{code}\nError: {result.error_in_exec}")
            if result.error_in_exec:
                outputs.append(nbf.v4.new_output(
                    output_type='error',
                    ename=type(result.error_in_exec).__name__,
                    evalue=str(result.error_in_exec),
                    traceback=[str(result.error_in_exec)]
                ))
            cell.outputs = outputs
            cell.execution_count = exec_count
            exec_count += 1
            break
            
        cell.outputs = outputs
        cell.execution_count = exec_count
        exec_count += 1
        
    with open(output_path, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
        
    print(f"Successfully executed and saved to {output_path} with all outputs and figures.")

if __name__ == '__main__':
    if len(sys.argv) > 1:
        execute_and_save_notebook(sys.argv[1])
    else:
        execute_and_save_notebook('notebooks/01_exploratory_data_analysis.ipynb')
