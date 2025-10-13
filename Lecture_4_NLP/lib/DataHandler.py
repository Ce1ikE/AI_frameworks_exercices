import pandas as pd

class DataHandler:
    @staticmethod
    def clean_data(df: pd.DataFrame):
        print("before cleaning:" + "-" * 20)
        df.info()
        # ----------------------------------
        # Steps taken
        # 1) renaming columns for easier access
        # 2) selecting relevant columns
        # 3) cleaning data (removing NaNs, lowercasing categories)
        # 4) concatenating title and text
        # 5) resetting index

        # 1)
        df.rename(
            columns={
                'Timestamp': 'timestamp', 
                'Title of your ticket': 'title',
                'Write your ticket' : 'text',
                'Your ticket is about:': 'category'
            }, 
            inplace=True
        )

        # 2) (timestamp is not relevant for our task)
        df = df[['title', 'text', 'category']]
        # 3) 
        df.dropna(inplace=True)
        df['category'] = df['category'].str.lower()
        # 4)
        df['text'] = df['title'] + ". " + df['text']
        df.drop(columns=['title'], inplace=True)
        # 5)
        df.reset_index(drop=True, inplace=True)
        # ----------------------------------
        print("after cleaning:" + "-" * 20)
        df.info()
        return df
    
    @staticmethod
    def split_data(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        from sklearn.model_selection import train_test_split
        X = df['text']
        y = df['category']
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y
        )
        train_data = pd.concat([X_train, y_train], axis=1)
        test_data = pd.concat([X_test, y_test], axis=1)
        return train_data, test_data