import ast
import random
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


RAW_DATA_PATH = Path("offline_feature_store/raw/profiles.parquet")

PROCESSED_DIR = Path("offline_feature_store/features")
TRAIN_PATH = PROCESSED_DIR / "train_pairs.parquet"
VALIDATION_PATH = PROCESSED_DIR / "validation_pairs.parquet"
TEST_PATH = PROCESSED_DIR / "test_pairs.parquet"

RANDOM_STATE = 42
NEGATIVE_RATIO = 10


def parse_feature_set(value):
    """
    Преобразует значение из feature-поля в set.

    Поддерживает:
    - list
    - tuple
    - set
    - numpy array
    - строку со списком
    - пустое / None значение
    """
    if value is None:
        return set()

    if isinstance(value, set):
        return set(map(str, value))

    if isinstance(value, (list, tuple)):
        return set(map(str, value))

    # parquet часто читает list-like колонки как numpy.ndarray
    if hasattr(value, "tolist"):
        try:
            value = value.tolist()

            if isinstance(value, list):
                return set(map(str, value))

            if value is None:
                return set()

            return {str(value)}
        except Exception:
            return set()

    if isinstance(value, str):
        value = value.strip()

        if not value or value.lower() in {"none", "nan", "null"}:
            return set()

        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, (list, tuple, set)):
                return set(map(str, parsed))

            if parsed is None:
                return set()

            return {str(parsed)}
        except Exception:
            return {value}

    try:
        if pd.isna(value):
            return set()
    except Exception:
        pass

    return {str(value)}


def normalize_string(value):
    """Приводит строковое значение к единому виду."""
    if value is None:
        return None

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value in {"", "none", "nan", "null"}:
        return None

    return value


def get_email_domain(email):
    """Извлекает домен email."""
    email = normalize_string(email)

    if email is None:
        return None

    if "@" not in email:
        return None

    return email.split("@")[-1]


def get_feature_value(features, prefix):
    """
    Достаёт значение из набора feature-строк по префиксу.

    Например:
    features = {"geoid:123", "device:mobile"}
    prefix = "geoid:"
    result = "123"
    """
    if not isinstance(features, set):
        return None

    for item in features:
        item = str(item)

        if item.startswith(prefix):
            return item.split(":", 1)[-1]

    return None


def get_local_hour(features):
    """Достаёт local_hour из realtime_features."""
    value = get_feature_value(features, "local_hour:")

    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def first_non_empty(series):
    """Возвращает первое непустое значение из группы."""
    for value in series:
        if value is None:
            continue

        try:
            if pd.isna(value):
                continue
        except Exception:
            pass

        if str(value).strip().lower() in {"", "none", "nan", "null"}:
            continue

        return value

    return None


def prepare_profiles(profiles_df):
    """
    Подготавливает исходные профили к генерации пар.

    На выходе одна строка = один profile_id.
    """
    required_columns = [
        "profile_id",
        "entity_id",
        "created_at",
        "first_name",
        "phone",
        "sex",
        "email",
        "fs_features",
        "non_processing_features",
        "realtime_features",
    ]

    missing_columns = [
        column for column in required_columns
        if column not in profiles_df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    profiles_df = profiles_df.copy()

    profiles_df["profile_id"] = profiles_df["profile_id"].astype(str)
    profiles_df["entity_id"] = profiles_df["entity_id"].astype(str)

    profile_level_df = (
        profiles_df[required_columns]
        .groupby("profile_id", as_index=False)
        .agg(first_non_empty)
    )

    profile_level_df["created_at"] = pd.to_datetime(
        profile_level_df["created_at"],
        errors="coerce",
    )

    profile_level_df["fs_set"] = profile_level_df["fs_features"].apply(parse_feature_set)
    profile_level_df["fs_count"] = profile_level_df["fs_set"].apply(len)

    profile_level_df["non_processing_set"] = (
        profile_level_df["non_processing_features"].apply(parse_feature_set)
    )

    profile_level_df["realtime_set"] = (
        profile_level_df["realtime_features"].apply(parse_feature_set)
    )

    profile_level_df["first_event"] = profile_level_df["created_at"]
    profile_level_df["event_count"] = 1

    profile_level_df["first_name"] = (
        profile_level_df["first_name"].apply(normalize_string)
    )

    profile_level_df["phone"] = (
        profile_level_df["phone"].apply(normalize_string)
    )

    profile_level_df["sex"] = (
        profile_level_df["sex"].apply(normalize_string)
    )

    profile_level_df["email_domain"] = (
        profile_level_df["email"].apply(get_email_domain)
    )

    profile_level_df["geoid"] = profile_level_df["non_processing_set"].apply(
        lambda features: get_feature_value(features, "geoid:")
    )

    profile_level_df["np_device"] = profile_level_df["non_processing_set"].apply(
        lambda features: get_feature_value(features, "device:")
    )

    profile_level_df["local_hour_mean"] = profile_level_df["realtime_set"].apply(
        get_local_hour
    )

    entity_sizes = profile_level_df.groupby("entity_id").size()
    multi_entities = set(entity_sizes[entity_sizes > 1].index)

    profile_level_df["is_multi_entity"] = (
        profile_level_df["entity_id"].isin(multi_entities)
    )

    return profile_level_df


def calculate_pair_features(p1, p2):
    """Вычисляет признаки для пары профилей."""
    fs1 = p1["fs_set"]
    fs2 = p2["fs_set"]

    union = len(fs1 | fs2)

    feats = {
        "time_diff_hours": abs(
            (p1["first_event"] - p2["first_event"]).total_seconds()
        ) / 3600
        if pd.notna(p1["first_event"]) and pd.notna(p2["first_event"])
        else -1,
        "event_count_diff": abs(p1["event_count"] - p2["event_count"]),
        "fs_jaccard": len(fs1 & fs2) / union if union > 0 else 0,
        "fs_count_diff": abs(p1["fs_count"] - p2["fs_count"]),
    }

    comparison_fields = {
        "geoid": "same_geoid",
        "first_name": "same_first_name",
        "phone": "same_phone",
        "sex": "same_sex",
        "email_domain": "same_email_domain",
        "np_device": "same_np_device",
    }

    for field, same_name in comparison_fields.items():
        v1 = p1.get(field)
        v2 = p2.get(field)

        if pd.notna(v1) and pd.notna(v2):
            feats[same_name] = int(str(v1) == str(v2))
        else:
            feats[same_name] = -1

    if pd.notna(p1["local_hour_mean"]) and pd.notna(p2["local_hour_mean"]):
        diff = abs(p1["local_hour_mean"] - p2["local_hour_mean"])
        feats["local_hour_mean_diff"] = min(diff, 24 - diff)
    else:
        feats["local_hour_mean_diff"] = -1

    return feats


def create_pairs(profiles_df, include_ids=False, negative_ratio=10):
    """
    Создаёт positive и negative пары профилей.

    positive:
    - одинаковый entity_id

    negative:
    - разный entity_id
    """
    positive_pairs = []
    negative_pairs = []

    entity_sizes = profiles_df.groupby("entity_id").size()
    multi_entities = entity_sizes[entity_sizes > 1].index.tolist()

    for entity_id in multi_entities:
        entity_profiles = profiles_df[profiles_df["entity_id"] == entity_id]
        profile_ids = entity_profiles.index.tolist()

        for i in range(len(profile_ids)):
            for j in range(i + 1, len(profile_ids)):
                p1 = profiles_df.loc[profile_ids[i]]
                p2 = profiles_df.loc[profile_ids[j]]

                feats = calculate_pair_features(p1, p2)
                feats["label"] = 1

                if include_ids:
                    feats["profile1"] = p1["profile_id"]
                    feats["profile2"] = p2["profile_id"]

                positive_pairs.append(feats)

    n_negative = len(positive_pairs) * negative_ratio

    multi_profiles = profiles_df[profiles_df["is_multi_entity"]].index.tolist()
    single_profiles = profiles_df[~profiles_df["is_multi_entity"]].index.tolist()

    random.seed(RANDOM_STATE)
    negative_count = 0
    max_attempts = n_negative * 5

    for _ in range(max_attempts):
        if negative_count >= n_negative:
            break

        if random.random() < 0.5 and len(multi_profiles) >= 2:
            idx1, idx2 = random.sample(multi_profiles, 2)
        elif multi_profiles and single_profiles:
            idx1 = random.choice(multi_profiles)
            idx2 = random.choice(single_profiles)
        else:
            continue

        p1 = profiles_df.loc[idx1]
        p2 = profiles_df.loc[idx2]

        if p1["entity_id"] != p2["entity_id"]:
            feats = calculate_pair_features(p1, p2)
            feats["label"] = 0

            if include_ids:
                feats["profile1"] = p1["profile_id"]
                feats["profile2"] = p2["profile_id"]

            negative_pairs.append(feats)
            negative_count += 1

    positive_df = pd.DataFrame(positive_pairs)
    negative_df = pd.DataFrame(negative_pairs)

    return positive_df, negative_df


def split_profiles_by_entity(profiles_df):
    """
    Делит профили на train / validation / test по entity_id.

    Это защищает от data leakage:
    один и тот же entity_id не должен попасть в разные выборки.
    """
    entities = profiles_df["entity_id"].drop_duplicates()

    train_val_entities, test_entities = train_test_split(
        entities,
        test_size=0.15,
        random_state=RANDOM_STATE,
    )

    train_entities, validation_entities = train_test_split(
        train_val_entities,
        test_size=0.1765,
        random_state=RANDOM_STATE,
    )

    train_profiles = profiles_df[
        profiles_df["entity_id"].isin(train_entities)
    ].copy()

    validation_profiles = profiles_df[
        profiles_df["entity_id"].isin(validation_entities)
    ].copy()

    test_profiles = profiles_df[
        profiles_df["entity_id"].isin(test_entities)
    ].copy()

    return train_profiles, validation_profiles, test_profiles


def build_pair_dataset(profiles_df, include_ids):
    positive_df, negative_df = create_pairs(
        profiles_df=profiles_df,
        include_ids=include_ids,
        negative_ratio=NEGATIVE_RATIO,
    )

    pairs_df = pd.concat(
        [positive_df, negative_df],
        ignore_index=True,
    )

    pairs_df = pairs_df.sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    return pairs_df, positive_df, negative_df


def main():
    print("1. Reading raw data")
    profiles_df = pd.read_parquet(RAW_DATA_PATH)

    print(f"Raw profiles shape: {profiles_df.shape}")

    print("\n2. Preparing profile-level dataset")
    profiles_df = prepare_profiles(profiles_df)

    print(f"Prepared profiles shape: {profiles_df.shape}")
    print(f"Unique profiles: {profiles_df['profile_id'].nunique()}")
    print(f"Unique entities: {profiles_df['entity_id'].nunique()}")

    print("\n3. Splitting profiles by entity_id")
    train_profiles, validation_profiles, test_profiles = split_profiles_by_entity(
        profiles_df
    )

    print(f"Train profiles: {train_profiles.shape}")
    print(f"Validation profiles: {validation_profiles.shape}")
    print(f"Test profiles: {test_profiles.shape}")

    print("\n4. Creating train pairs")
    train_pairs, train_pos, train_neg = build_pair_dataset(
        train_profiles,
        include_ids=False,
    )

    print(f"Train positive pairs: {len(train_pos)}")
    print(f"Train negative pairs: {len(train_neg)}")
    print(f"Train total pairs: {len(train_pairs)}")

    print("\n5. Creating validation pairs")
    validation_pairs, validation_pos, validation_neg = build_pair_dataset(
        validation_profiles,
        include_ids=True,
    )

    print(f"Validation positive pairs: {len(validation_pos)}")
    print(f"Validation negative pairs: {len(validation_neg)}")
    print(f"Validation total pairs: {len(validation_pairs)}")

    print("\n6. Creating test pairs")
    test_pairs, test_pos, test_neg = build_pair_dataset(
        test_profiles,
        include_ids=True,
    )

    print(f"Test positive pairs: {len(test_pos)}")
    print(f"Test negative pairs: {len(test_neg)}")
    print(f"Test total pairs: {len(test_pairs)}")

    print("\n7. Saving processed datasets")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    train_pairs.to_parquet(TRAIN_PATH, index=False)
    validation_pairs.to_parquet(VALIDATION_PATH, index=False)
    test_pairs.to_parquet(TEST_PATH, index=False)

    print(f"Saved train: {TRAIN_PATH}")
    print(f"Saved validation: {VALIDATION_PATH}")
    print(f"Saved test: {TEST_PATH}")

    print("\nDone.")


if __name__ == "__main__":
    main()