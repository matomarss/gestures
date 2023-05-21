from files_organization import dump_to_json, dump_object, load_from_json


def load_kernel_experiment_results_for_pca_table(pca):
    """
        For the given pca, load all corresponding results for every kernel and preprocessing for a table generation
    """
    results = load_from_json("test_svm_kernels", "SVM_kernel_test_n=20")
    processed_results = {}
    for kernel in results.keys():
        processed_results[kernel] = {}
        for rec in results.get(kernel):
            if rec.get("pca") != pca:
                continue
            prep = ""
            for key in rec.keys():
                if key != "validation_accuracy":
                    prep += "&"+str(rec.get(key))
            processed_results[kernel][prep] = rec["validation_accuracy"]
    return processed_results


def load_preprocessing_experiment_results(n, mod):
    """
    Load all results from the preprocessing experiment
    """
    results = {}

    direc = "test_pca_n=" + str(n)
    filename_no_pca = mod.get_name() + "test_with_NO_PCA_n=" + str(n)
    filename_pca = mod.get_name() + "test_with_PCA_n=" + str(n)
    data = load_from_json(direc, filename_pca)
    results[n*18] = data.get("all")
    for n_comp in data.keys():
        if n_comp == "all":
            continue
        results[int(n_comp)] = data.get(n_comp)

    data = load_from_json(direc, filename_no_pca)
    results[-1] = data.get("NO_PCA")

    results = dict(sorted(results.items()))

    return results


def find_highest_accuracy_at_preprocessing_experiment(mod, n):
    """
    Finds the highest validation accuracy achieved by the given model at the given sequence length from all preprocessing combinations
    """
    best_result = {"validation_accuracy": 0}
    direc = "test_pca_n=" + str(n)
    filename_no_pca = mod.get_name() + "test_with_NO_PCA_n=" + str(n)
    filename_pca = mod.get_name() + "test_with_PCA_n=" + str(n)
    data = load_from_json(direc, filename_pca)
    for n_comp in data.keys():
        records = data[n_comp]
        for i in range(len(records)):
            val_acc = records[i].get("validation_accuracy")
            if val_acc > best_result.get("validation_accuracy"):
                best_result["validation_accuracy"] = val_acc
                best_result["n"] = n
                best_result["preprocessing"] = records[i].get("preprocessing")
                best_result["scaler"] = records[i].get("scaler")
                best_result["pca"] = n_comp
    data = load_from_json(direc, filename_no_pca)
    records = data.get("NO_PCA")
    for rec in records:
        val_acc = rec.get("validation_accuracy")
        if val_acc > best_result.get("validation_accuracy"):
            best_result["validation_accuracy"] = val_acc
            best_result["n"] = n
            best_result["preprocessing"] = rec.get("preprocessing")
            best_result["scaler"] = rec.get("scaler")
            best_result["pca"] = "None"
    return best_result


def load_preprocessing_experiment_average_accuracies(n, mod):
    """
        For each component left after PCA or for NO PCA load average validation accuracy achieved (for a given model and sequence length)
    """
    accuracies = {}

    direc = "test_pca_n="+str(n)
    filename_no_pca = mod.get_name() + "test_with_NO_PCA_n="+str(n)
    filename_pca = mod.get_name() + "test_with_PCA_n="+str(n)

    data = load_from_json(direc, filename_pca)
    for n_comp in data.keys():
        if n_comp == "all":
            acc_key = n*18
        else:
            acc_key = int(n_comp)
        records = data[n_comp]
        sum_val_acc = 0
        for i in range(len(records)):
            sum_val_acc += records[i].get("validation_accuracy")
        val_acc = sum_val_acc/len(records)
        accuracies[acc_key] = val_acc

    data = load_from_json(direc, filename_no_pca)
    records = data.get("NO_PCA")
    sum_val_acc_no_scaler = 0
    sum_val_acc_scaler = 0
    for rec in records:
        if rec.get("scaler") == "None":
            sum_val_acc_no_scaler += rec.get("validation_accuracy")
        else:
            sum_val_acc_scaler += rec.get("validation_accuracy")
    val_acc_no_scaler = sum_val_acc_no_scaler/2
    val_acc_scaler = sum_val_acc_scaler/4
    accuracies[-2] = val_acc_no_scaler
    accuracies[-1] = val_acc_scaler

    return accuracies


def load_preprocessing_experiment_best_accuracies(n, mod):
    """
        For each component left after PCA or for NO PCA load best validation accuracy achieved (for a given model and sequence length)
    """
    accuracies = {}

    direc = "test_pca_n=" + str(n)
    filename_no_pca = mod.get_name() + "test_with_NO_PCA_n=" + str(n)
    filename_pca = mod.get_name() + "test_with_PCA_n=" + str(n)
    data = load_from_json(direc, filename_pca)
    for n_comp in data.keys():
        if n_comp == "all":
            acc_key = n*18
        else:
            acc_key = int(n_comp)
        records = data[n_comp]
        accuracies[acc_key] = 0
        for i in range(len(records)):
            val_acc = records[i].get("validation_accuracy")
            if val_acc > accuracies.get(acc_key):
                accuracies[acc_key] = val_acc
    data = load_from_json(direc, filename_no_pca)
    records = data.get("NO_PCA")
    accuracies[-1] = 0
    for rec in records:
        val_acc = rec.get("validation_accuracy")
        if val_acc > accuracies.get(-1):
            accuracies[-1] = val_acc

    return accuracies


def load_best_accuracies_after_hyper_parameter_search(mod):
    """
        For the given model, load the best validation accuracy achieved for every sequence length for its best set of hyperparameters
    """
    results = {}
    direc = "test_hyper_parameters"
    for n in [1, 10, 20, 40]:
        filename = mod.get_name() + "_hyper_par_test_n=" + str(n)
        data = load_from_json(direc, filename)
        results[n] = data.get("best_parameters").get("best_score")
    return results

