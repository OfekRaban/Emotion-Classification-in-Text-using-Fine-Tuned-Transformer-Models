from complete_pipeline_gpu import run_single_experiment, ExperimentConfig
from comparer import ModelComparer   # אם זה אצלך בקובץ אחר

comparer = ModelComparer()

experiments = [
    # ===== Baseline =====
    dict(
        experiment_name="lstm_baseline",
        model_type="lstm",
        rnn_units=128,
        dropout=0.2,
        batch_size=32,
        learning_rate=1e-3
    ),

    # ===== Units =====
    dict(experiment_name="lstm_units_64",  rnn_units=64),
    dict(experiment_name="lstm_units_256", rnn_units=256),

    # ===== Dropout =====
    dict(experiment_name="lstm_dropout_01", dropout=0.1),
    dict(experiment_name="lstm_dropout_03", dropout=0.3),

    # ===== Batch size =====
    dict(experiment_name="lstm_batch_16", batch_size=16),
    dict(experiment_name="lstm_batch_64", batch_size=64),

    # ===== Architecture =====
    dict(experiment_name="gru_128", model_type="gru", rnn_units=128),
    dict(experiment_name="gru_256", model_type="gru", rnn_units=256),
]

base_config = ExperimentConfig()

for exp in experiments:
    print(f"\n=== Running {exp['experiment_name']} ===")

    config = ExperimentConfig(**base_config.to_dict())

    for k, v in exp.items():
        setattr(config, k, v)

    history, metrics = run_single_experiment(config)

    comparer.add_experiment(
        name=config.experiment_name,
        config=config,
        history=history,
        metrics=metrics
    )

# שמירת טבלת השוואה
comparer.create_comparison_table()
comparer.save_comparison("results/model_comparison.csv")
