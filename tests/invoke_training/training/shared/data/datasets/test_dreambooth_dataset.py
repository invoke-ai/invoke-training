from invoke_training.training.shared.data.datasets.dreambooth_dataset import (
    DreamBoothDataset,
)


def test_dreambooth_dataset_instance_only():
    """Test the behavior of DreamBoothDataset with only an instance_dataset i.e. no class_dataset."""
    instance_prompt = "test instance prompt"
    # We use a list as a mock dataset. It satisfies the torch.utils.data.Dataset interface.
    mock_instance_dataset = [{"field1": 1}]

    dataset = DreamBoothDataset(mock_instance_dataset, instance_prompt, shuffle=True)

    output = dataset[0]

    expected = {
        "field1": 1,
        "caption": instance_prompt,
        "loss_weight": 1.0,
    }
    assert output == expected


def test_dreambooth_dataset_with_class_dataset():
    """Test DreamBoothDataset with both an instance_dataset and class_dataset."""
    instance_prompt = "test instance prompt"
    class_prompt = "test class prompt"
    # We use a list as a mock dataset. It satisfies the torch.utils.data.Dataset interface.
    mock_instance_dataset = [{"field1": 1}]
    mock_class_dataset = [{"field1": 2}, {"field1": 3}]

    dataset = DreamBoothDataset(
        instance_dataset=mock_instance_dataset,
        instance_prompt=instance_prompt,
        class_dataset=mock_class_dataset,
        class_prompt=class_prompt,
        prior_preservation_loss_weight=0.5,
        balance_datasets=False,
        shuffle=False,
    )

    output = [o for o in dataset]

    # When shuffle=False, all instance_dataset elements should be returned first, followed by all class_dataset
    # elements.
    expected = [
        {
            "field1": 1,
            "caption": instance_prompt,
            "loss_weight": 1.0,
        },
        {
            "field1": 2,
            "caption": class_prompt,
            "loss_weight": 0.5,
        },
        {
            "field1": 3,
            "caption": class_prompt,
            "loss_weight": 0.5,
        },
    ]
    assert output == expected


def test_dreambooth_dataset_balance_datasets_class_is_longer():
    """Test DreamBoothDataset dataset balancing when the class_dataset is larger than the instance_dataset."""
    instance_prompt = "test instance prompt"
    class_prompt = "test class prompt"
    # We use a list as a mock dataset. It satisfies the torch.utils.data.Dataset interface.
    mock_instance_dataset = [{"field1": 1}]
    mock_class_dataset = [{"field1": 2}, {"field1": 3}]

    dataset = DreamBoothDataset(
        instance_dataset=mock_instance_dataset,
        instance_prompt=instance_prompt,
        class_dataset=mock_class_dataset,
        class_prompt=class_prompt,
        prior_preservation_loss_weight=0.5,
        balance_datasets=True,
        shuffle=False,
    )

    output = [o for o in dataset]

    # When shuffle=False, all instance_dataset elements should be returned first, followed by all class_dataset
    # elements.
    expected = [
        {
            "field1": 1,
            "caption": instance_prompt,
            "loss_weight": 1.0,
        },
        # The instance_dataset element is repeated to match the size of the class_dataset.
        {
            "field1": 1,
            "caption": instance_prompt,
            "loss_weight": 1.0,
        },
        {
            "field1": 2,
            "caption": class_prompt,
            "loss_weight": 0.5,
        },
        {
            "field1": 3,
            "caption": class_prompt,
            "loss_weight": 0.5,
        },
    ]
    assert output == expected


def test_dreambooth_dataset_balance_datasets_instance_is_longer():
    """Test DreamBoothDataset dataset balancing when the instance_dataset is larger than the class_dataset."""
    instance_prompt = "test instance prompt"
    class_prompt = "test class prompt"
    # We use a list as a mock dataset. It satisfies the torch.utils.data.Dataset interface.
    mock_instance_dataset = [{"field1": 1}, {"field1": 2}]
    mock_class_dataset = [{"field1": 3}]

    dataset = DreamBoothDataset(
        instance_dataset=mock_instance_dataset,
        instance_prompt=instance_prompt,
        class_dataset=mock_class_dataset,
        class_prompt=class_prompt,
        prior_preservation_loss_weight=0.5,
        balance_datasets=True,
        shuffle=False,
    )

    output = [o for o in dataset]

    # When shuffle=False, all instance_dataset elements should be returned first, followed by all class_dataset
    # elements.
    expected = [
        {
            "field1": 1,
            "caption": instance_prompt,
            "loss_weight": 1.0,
        },
        {
            "field1": 2,
            "caption": instance_prompt,
            "loss_weight": 1.0,
        },
        {
            "field1": 3,
            "caption": class_prompt,
            "loss_weight": 0.5,
        },
        # The class_dataset element is repeated to match the size of the instance_dataset.
        {
            "field1": 3,
            "caption": class_prompt,
            "loss_weight": 0.5,
        },
    ]
    assert output == expected


def test_dreambooth_dataset_shuffle():
    """Test DreamBoothDataset shuffling."""
    instance_prompt = "test instance prompt"
    class_prompt = "test class prompt"
    # We use a list as a mock dataset. It satisfies the torch.utils.data.Dataset interface.
    mock_instance_dataset = [{"field1": 1}, {"field1": 2}]
    mock_class_dataset = [{"field1": 3}]

    dataset = DreamBoothDataset(
        instance_dataset=mock_instance_dataset,
        instance_prompt=instance_prompt,
        class_dataset=mock_class_dataset,
        class_prompt=class_prompt,
        prior_preservation_loss_weight=0.5,
        balance_datasets=True,
        shuffle=True,
    )

    output = [o for o in dataset]

    # A sorted list of expected return values.
    expected = [
        {
            "field1": 1,
            "caption": instance_prompt,
            "loss_weight": 1.0,
        },
        {
            "field1": 2,
            "caption": instance_prompt,
            "loss_weight": 1.0,
        },
        {
            "field1": 3,
            "caption": class_prompt,
            "loss_weight": 0.5,
        },
        # The class_dataset element is repeated to match the size of the instance_dataset.
        {
            "field1": 3,
            "caption": class_prompt,
            "loss_weight": 0.5,
        },
    ]

    sorted_output = sorted(output, key=lambda x: x["field1"])

    assert expected == sorted_output
