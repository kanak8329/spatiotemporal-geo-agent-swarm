import subprocess
import time
import os

print("=" * 60)
print(" 🚀 INITIATING SELF-RUNNING MULTIMODAL PIPELINE")
print("=" * 60)


def run_script(script_name):
    if not os.path.exists(script_name):
        print(f"[!] Error: Cannot find {script_name}")
        return False
    print(f"\n▶️ Executing: {script_name}...")
    start_time = time.time()
    result = subprocess.run(["python", script_name], text=True)
    if result.returncode == 0:
        print(f"✅ Completed in {time.time() - start_time:.1f} seconds.")
        return True
    else:
        print(f"❌ {script_name} failed! Pipeline halted.")
        return False


def run_pipeline():
    # 1. Extract, Fuse, and V&V 2.2M points from Dropzone to Validation Vault
    if not run_script("unified_data_extractor.py"): return

    # 2. Re-aggregate the fresh data for Streamlit UI speed
    if not run_script("aggregate_data.py"): return

    # 3. Retrain the Physical World Model (XGBoost)
    if not run_script("ml_predictor.py"): return

    # 4. Retrain the Agentic Reinforcement Learning Policy (PPO)
    if not run_script("rl_trainer.py"): return

    print("\n" + "=" * 60)
    print(" 🌟 MLOPS PIPELINE COMPLETE! Streamlit Dashboard is ready.")
    print(" 🌊 Run `streamlit run app_v2.py` to view the thesis app.")
    print("=" * 60)


if __name__ == "__main__":
    run_pipeline()