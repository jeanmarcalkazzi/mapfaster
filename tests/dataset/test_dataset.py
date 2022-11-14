import hashlib


def test_dataset_analytics_integrity():
    with open("dataset/dataframes/maps_analytics.csv", "r") as maps_analytics_file:
        assert (
            hashlib.md5(maps_analytics_file.read().encode("utf-8")).hexdigest()
            == "162777492ea753539b222261fe583252"
        )
