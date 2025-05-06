#!/usr/bin/env nextflow

/*
 * Pipeline parameters
 */
params.train_out      = 'training_results'  // Output directory for training
params.processed_data = 'data/processed/*.h5ad' // Accepts multiple processed AnnData files

// Parameters for train_models.py (adjust these as needed)
params.neighbors      = 3
params.hidden_size    = 64
params.lr             = 0.01
params.dr             = 0.5
params.epochs         = 20
params.gridsearch     = false

/*
 * Process: TrainModels
 * This process runs model training for each input file.
 */
process TrainModels {

    input:
    // Accept multiple processed AnnData files as input
    path processed_data from file(params.processed_data)

    output:
    // Output directory for training results
    path "${params.train_out}/train_${processed_data.baseName}.log"

    script:
    """
    mkdir -p ${params.train_out}
    python classification.py \\
        --input ${processed_data} \\
        --output ${params.train_out} \\
        --neighbors ${params.neighbors} \\
        --hidden-size ${params.hidden_size} \\
        --lr ${params.lr} \\
        --dr ${params.dr} \\
        --epochs ${params.epochs} \\
        $( [ "${params.gridsearch}" = "true" ] && echo "--gridsearch" ) > ${params.train_out}/train_${processed_data.baseName}.log
    """
}

/*
 * Define the workflow
 */
workflow {
    // Run TrainModels process for each input file
    TrainModels()
}