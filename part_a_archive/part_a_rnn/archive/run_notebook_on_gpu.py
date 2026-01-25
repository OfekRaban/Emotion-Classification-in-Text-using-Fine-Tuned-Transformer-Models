#!/usr/bin/env python3
"""
Execute complete_pipeline.ipynb on GPU 3
This script runs the notebook and saves results back to the notebook file.
"""

import os
import sys
import json
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
from datetime import datetime

def setup_gpu(gpu_id=None):
    """Configure GPU settings"""
    # Use environment variable if set, otherwise default to 3
    if gpu_id is None:
        gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '3')

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    print(f"=" * 70)
    print(f"GPU Configuration")
    print(f"=" * 70)
    print(f"CUDA_VISIBLE_DEVICES: {gpu_id}")
    print(f"TF_FORCE_GPU_ALLOW_GROWTH: true")
    print(f"=" * 70)
    print()

    # Force flush
    import sys
    sys.stdout.flush()

def verify_gpu():
    """Verify GPU is available"""
    try:
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✓ TensorFlow detected {len(gpus)} GPU(s)")
            for gpu in gpus:
                print(f"  - {gpu}")
            # Configure memory growth
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Memory growth enabled")
        else:
            print(f"✗ WARNING: No GPU detected by TensorFlow")
            print(f"  The notebook will run on CPU (slower)")
        print()
    except ImportError:
        print(f"✗ TensorFlow not imported yet")
        print()

def execute_notebook(notebook_path, output_path=None, timeout=7200):
    """
    Execute a Jupyter notebook

    Args:
        notebook_path: Path to input notebook
        output_path: Path to save executed notebook (default: overwrites input)
        timeout: Timeout in seconds (default: 7200 = 2 hours)
    """
    if output_path is None:
        output_path = notebook_path

    print(f"=" * 70)
    print(f"Notebook Execution")
    print(f"=" * 70)
    print(f"Input:   {notebook_path}")
    print(f"Output:  {output_path}")
    print(f"Timeout: {timeout} seconds ({timeout/60:.0f} minutes)")
    print(f"=" * 70)
    print()

    # Load notebook
    print(f"Loading notebook...")
    with open(notebook_path) as f:
        nb = nbformat.read(f, as_version=4)

    print(f"✓ Loaded: {len(nb.cells)} cells")
    print()

    # Count sections
    sections = [c for c in nb.cells if c.cell_type == 'markdown' and
                any('Section' in line for line in (c.source if isinstance(c.source, list) else [c.source]))]
    print(f"✓ Found: {len(sections)} sections")
    print()

    # Create executor
    print(f"Starting execution...")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 70)
    print()

    ep = ExecutePreprocessor(
        timeout=timeout,
        kernel_name='python3',
        allow_errors=False  # Stop on errors
    )

    # Execute
    start_time = datetime.now()
    try:
        ep.preprocess(nb, {'metadata': {'path': 'notebooks/'}})
        success = True
    except Exception as e:
        print()
        print(f"=" * 70)
        print(f"ERROR DURING EXECUTION")
        print(f"=" * 70)
        print(f"{type(e).__name__}: {e}")
        print()
        success = False
        # Save partial results anyway

    end_time = datetime.now()
    duration = end_time - start_time

    # Save executed notebook
    print()
    print(f"=" * 70)
    print(f"Saving executed notebook...")
    with open(output_path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"✓ Saved: {output_path}")
    print(f"=" * 70)
    print()

    # Summary
    print(f"=" * 70)
    print(f"EXECUTION SUMMARY")
    print(f"=" * 70)
    print(f"Status:   {'✓ SUCCESS' if success else '✗ FAILED'}")
    print(f"Duration: {duration}")
    print(f"Start:    {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End:      {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output:   {output_path}")
    print(f"=" * 70)

    return success

def main():
    """Main execution"""
    print()
    print(f"=" * 70)
    print(f"COMPLETE PIPELINE - GPU EXECUTION")
    print(f"=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"=" * 70)
    print()

    # Setup GPU 3
    setup_gpu(gpu_id=3)

    # Verify GPU
    verify_gpu()

    # Notebook paths
    notebook_path = 'notebooks/complete_pipeline.ipynb'
    output_path = 'notebooks/complete_pipeline_executed.ipynb'

    # Check if notebook exists
    if not os.path.exists(notebook_path):
        print(f"✗ ERROR: Notebook not found: {notebook_path}")
        sys.exit(1)

    # Execute
    success = execute_notebook(
        notebook_path=notebook_path,
        output_path=output_path,
        timeout=7200  # 2 hours max
    )

    if success:
        print()
        print(f"✓ NOTEBOOK EXECUTION COMPLETED SUCCESSFULLY")
        print()
        print(f"Results saved to:")
        print(f"  - {output_path}")
        print()
        print(f"Generated files (check these directories):")
        print(f"  - results/")
        print(f"  - logs/")
        print(f"  - saved_models/")
        print()
        sys.exit(0)
    else:
        print()
        print(f"✗ NOTEBOOK EXECUTION FAILED")
        print(f"Check the output notebook for error details:")
        print(f"  - {output_path}")
        print()
        sys.exit(1)

if __name__ == '__main__':
    main()
