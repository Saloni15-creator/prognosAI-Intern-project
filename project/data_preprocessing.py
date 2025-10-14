import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse
import os
import joblib

class DataPreprocessor:
    def __init__(self, window_size=30, scaler_type='standard'):
        self.window_size = window_size
        self.scaler_type = scaler_type
        self.scaler = None
        self.feature_cols = None

def load_data(self, file_path, dataset_type='train'):
        column_names = ['engine_id', 'cycle'] + \
                       [f'op_setting_{i}' for i in range(1, 4)] + \
                       [f'sensor_{i}' for i in range(1, 22)]
        if file_path.endswith('.txt'):
            df = pd.read_csv(file_path, sep=r'\s+', header=None, names=column_names)
        else:
            df = pd.read_csv(file_path)
        print(f"Loaded {dataset_type} data: {df.shape}")
        return df

def calculate_rul(self, df, dataset_type='train', rul_file_path=None):
        df_rul = df.copy()
        if dataset_type == 'train':
            max_cycles = df_rul.groupby('engine_id')['cycle'].max().reset_index()
            max_cycles.columns = ['engine_id', 'max_cycle']
            df_rul = df_rul.merge(max_cycles, on='engine_id', how='left')
            df_rul['RUL'] = df_rul['max_cycle'] - df_rul['cycle']
            df_rul.drop('max_cycle', axis=1, inplace=True)
        elif dataset_type == 'test' and rul_file_path and os.path.exists(rul_file_path):
            rul_true = pd.read_csv(rul_file_path, header=None, names=['RUL_true'])
            rul_true['engine_id'] = rul_true.index + 1
            last_cycles = df_rul.groupby('engine_id')['cycle'].max().reset_index()
            last_cycles = last_cycles.merge(rul_true, on='engine_id', how='left')
            df_rul = df_rul.merge(last_cycles[['engine_id', 'cycle', 'RUL_true']],
                                  on='engine_id', how='left', suffixes=('', '_last'))
            df_rul['RUL'] = df_rul['RUL_true'] + (df_rul['cycle_last'] - df_rul['cycle'])
            df_rul.drop(['RUL_true', 'cycle_last'], axis=1, inplace=True)
        else:
            df_rul['RUL'] = 0
        return df_rul

def feature_engineering(self, df):
        df_features = df.copy()
        sensor_cols = [col for col in df_features.columns if 'sensor_' in col]
        constant_sensors = [col for col in sensor_cols if df_features[col].std() == 0]
        if constant_sensors:
            df_features.drop(constant_sensors, axis=1, inplace=True)
            sensor_cols = [col for col in sensor_cols if col not in constant_sensors]

        for col in sensor_cols:
            grp = df_features.groupby('engine_id')[col]
            df_features[f'{col}_rollmean5'] = grp.rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
            df_features[f'{col}_rollstd5'] = grp.rolling(5, min_periods=1).std().reset_index(level=0, drop=True).fillna(0)

        return df_features

def normalize_features(self, df):
        if self.scaler_type == 'none':
            return df
        df_norm = df.copy()
        exclude_cols = ['engine_id', 'cycle', 'RUL']

        feature_cols_path = os.path.join('processed_data', 'train', 'feature_columns.txt')
        if os.path.exists(feature_cols_path):
            with open(feature_cols_path, 'r') as f:
                trained_features = [line.strip() for line in f.readlines()]
            for col in trained_features:
                if col not in df_norm.columns:
                    df_norm[col] = 0.0
            extra_cols = [col for col in df_norm.columns if col not in trained_features and col not in exclude_cols]
            if extra_cols:
                df_norm.drop(extra_cols, axis=1, inplace=True)
            self.feature_cols = trained_features
        else:
            self.feature_cols = [col for col in df_norm.columns if col not in exclude_cols]

        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler type: {self.scaler_type}")

        df_norm[self.feature_cols] = self.scaler.fit_transform(df_norm[self.feature_cols])
        return df_norm

def generate_sequences(self, df):
        sequences = []
        metadata = []
        df_sorted = df.sort_values(['engine_id', 'cycle']).reset_index(drop=True)
        for engine_id in df_sorted['engine_id'].unique():
            engine_data = df_sorted[df_sorted['engine_id'] == engine_id]
            features = engine_data[self.feature_cols].values
            cycles = engine_data['cycle'].values
            ruls = engine_data['RUL'].values
            for i in range(self.window_size - 1, len(engine_data)):
                seq = features[i - self.window_size + 1:i + 1]
                sequences.append(seq)
                metadata.append({'engine_id': engine_id, 'cycle': cycles[i], 'RUL': ruls[i], 'sequence_idx': len(sequences) - 1})
        sequences = np.array(sequences)
        metadata_df = pd.DataFrame(metadata)
        return sequences, metadata_df

def save_processed_data(self, sequences, metadata_df, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        seq_path = os.path.join(output_dir, 'sequences.npy')
        meta_path = os.path.join(output_dir, 'metadata.csv')
        np.save(seq_path, sequences)
        metadata_df.to_csv(meta_path, index=False)
        if self.scaler:
            scaler_path = os.path.join(output_dir, 'scaler.pkl')
            joblib.dump(self.scaler, scaler_path)
        feature_path = os.path.join(output_dir, 'feature_columns.txt')
        with open(feature_path, 'w') as f:
            for col in self.feature_cols:
                f.write(col + '\n')
        return seq_path, meta_path

def process_data(self, input_file, dataset_type='train', rul_file=None, output_dir='processed_data'):
        df = self.load_data(input_file, dataset_type)
        df = self.calculate_rul(df, dataset_type, rul_file)
        df = self.feature_engineering(df)
        df = self.normalize_features(df)
        sequences, metadata_df = self.generate_sequences(df)
        return self.save_processed_data(sequences, metadata_df, output_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', required=True, help='Path to input data file (relative to data/raw/)')
    parser.add_argument('--dataset_type', default='train', choices=['train', 'test'], help='Dataset type')
    parser.add_argument('--rul_file', default=None, help='Path to RUL file (relative to data/raw/), only for test')
    parser.add_argument('--window_size', type=int, default=30, help='Window size for sequences')
    parser.add_argument('--scaler_type', default='standard', choices=['standard', 'minmax', 'none'], help='Scaler type')
    args = parser.parse_args()

    input_path = os.path.join('data', 'raw', args.input_file)
    out_dir = os.path.join('processed_data', args.dataset_type)
    rul_path = os.path.join('data', 'raw', args.rul_file) if args.rul_file else None

    dp = DataPreprocessor(window_size=args.window_size, scaler_type=args.scaler_type)
    dp.process_data(input_path, args.dataset_type, rul_path, out_dir)
    print(f"Processed {args.dataset_type} data saved to {out_dir}")

if __name__ == '__main__':

    dp = DataPreprocessor(window_size=30, scaler_type='standard')
    dp.process_data(
        input_file='data/raw/train_FD001.txt',
        dataset_type='train',
        rul_file=None,
        output_dir='processed_data/train'
    )
    dp.process_data(
        input_file='data/raw/test_FD001.txt',
        dataset_type='test',
        rul_file='data/raw/RUL_FD001.txt',
        output_dir='processed_data/test'
    )
    print("Data preprocessing completed successfully!")
