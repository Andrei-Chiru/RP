from collections import defaultdict


def train(
        epochs,
        metrics_dict,
        ds_train,
        ds_test_clean,
        train_step,
        ds_test_poisoned = None,
        test_step_clean= None,
        test_step_poisoned = None,
        csv_path=None,
        scheduled_parameters=defaultdict(lambda: {})
):
    """
    Args:
        epochs: int, number of training epochs.
        metrics_dict: dict, {"metrics_label": tf.keras.metrics instance}.
        ds_train: iterable dataset, e.g. using tf.data.Dataset.
        ds_test: iterable dataset, e.g. using tf.data.Dataset.
        train_step: callable function. the arguments passed to the function
            are the itered elements of ds_train.
        test_step: callable function. the arguments passed to the function
            are the itered elements of ds_test.
        csv_path: (optional) path to create a csv file, to save the metrics.
        scheduled_parameters: (optional) a dictionary that returns kwargs for
            the train_step and test_step functions, for each epoch.
            Call using scheduled_parameters[epoch].
    """
    template = "Epoch {}"
    train_loss = []
    test_loss = []
    train_acc = []
    clean_acc = []
    asr = []
    for metrics_label in metrics_dict.keys():
        template += ", %s: {:.4f}" % metrics_label
    if csv_path is not None:
        csv_file = open(csv_path, "w+")
        headers = ",".join(["Epoch"] + list(metrics_dict.keys()))
        csv_template = ",".join(["{}" for _ in range(len(metrics_dict) + 1)])
        csv_file.write(headers + "\n")

    for epoch in range(epochs):
        for metrics in metrics_dict.values():
            metrics.reset_state()

        for batch_elements in ds_train:
            train_step(*batch_elements, **scheduled_parameters[epoch])
        for batch_elements in ds_test_clean:
            test_step_clean(*batch_elements, **scheduled_parameters[epoch])
        if ds_test_poisoned is not None:
            for batch_elements in ds_test_poisoned:
                test_step_poisoned(*batch_elements, **scheduled_parameters[epoch])
        metrics_results = [metrics.result() for metrics in metrics_dict.values()]
        train_loss.append(metrics_dict['train_loss'].result().numpy())
        test_loss.append(metrics_dict['test_loss'].result().numpy())
        train_acc.append(metrics_dict['train_accuracy'].result().numpy())
        clean_acc.append(metrics_dict['clean_accuracy'].result().numpy())
        asr.append(metrics_dict['attack_success_rate'].result().numpy())
        print(template.format(epoch, *metrics_results))
        if csv_path is not None:
            csv_file.write(csv_template.format(epoch, *metrics_results) + "\n" + "\n")
            csv_file.flush()
    if csv_path is not None:
        csv_file.close()

    return train_loss, test_loss, train_acc, clean_acc, asr