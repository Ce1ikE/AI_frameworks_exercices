import pandas as pd
import pycountry

class DataHandler:
    @staticmethod
    def clean_data(df: pd.DataFrame):
        print("before cleaning:" + "-" * 20)
        df.info()

        #### **Data description**
        # - **country:** Name of the country.
        # - **child_mort:** Death of children under 5 years of age per 1000 live births.
        # - **exports:** Exports of goods and services per capita. Given as %age of the GDP per capita.
        # - **health:** Total health spending per capita. Given as %age of GDP per capita.
        # - **imports:** Imports of goods and services per capita. Given as %age of the GDP per capita.
        # - **income:** Net income per person.
        # - **inflation:** The measurement of the annual growth rate of the Total GDP.
        # - **life_expec:** The average number of years a new born child would live if the current mortality patterns are to remain the same.
        # - **total_fer:** The number of children that would be born to each woman if the current age-fertility rates remain the same.
        # - **gdpp:** The GDP per capita. Calculated as the Total GDP divided by the total population.

        # ----------------------------------
        # Steps taken
        # 1) normalize country names
        df = DataHandler.normalize_country_names(df)
        # ----------------------------------
        print("after cleaning:" + "-" * 20)
        df.info()
        
        return df


    @staticmethod
    def get_numerical_and_categorical_features(df: pd.DataFrame):
        col = list(df.columns)
        col.remove('country')
        categorical_features = ['country']
        numerical_features = [*col] 

        print("\nCategorical Features:", categorical_features)
        print("Numerical Features:", numerical_features)
        return numerical_features, categorical_features
    
    @staticmethod
    def compress_country_data(df: pd.DataFrame):
        # - If we needed to reduce this dataset to just 3 features (without removing any). How would you perform that? 
        # (Just using the knowledge and insights that you got from the graphs)
        # for example, we could combine child_mort and life_expec into a single feature representing health outcomes,
        # and keep income and gdpp as separate features.
        compressed_data = pd.DataFrame()
        # for health we use child_mort , life_expec , health and total_fer
        compressed_data['Health'] = df['child_mort'] / df['child_mort'].mean() + df['life_expec'] / df['life_expec'].mean() + df['health'] / df['health'].mean() + df['total_fer'] / df['total_fer'].mean()
        # for trade we use exports, and imports
        compressed_data['Trade'] = df['exports'] / df['exports'].mean() + df['imports'] / df['imports'].mean()
        # for finance we want to maximize gdpp, inflation, and income
        compressed_data['Finance'] = df['gdpp'] / df['gdpp'].mean() + df['inflation'] / df['inflation'].mean() + df['income'] / df['income'].mean()
        # set country as index
        compressed_data.index = df['country']
        print("\nCompressed Data (3 features):")
        print(compressed_data.head())
        compressed_data.info()
        return compressed_data

    @staticmethod
    def add_country_continent(df: pd.DataFrame):
        def get_continent(country_name):
            try:
                country = pycountry.countries.get(name=country_name)
                if not country:
                    country = pycountry.countries.search_fuzzy(country_name)[0]
                continent_code = pycountry.subdivisions.get(country_code=country.alpha_2).type
                continent_mapping = {
                    'Africa': 'Africa',
                    'Asia': 'Asia',
                    'Europe': 'Europe',
                    'North America': 'North America',
                    'Oceania': 'Oceania',
                    'South America': 'South America',
                    'Antarctica': 'Antarctica'
                }
                return continent_mapping.get(continent_code, 'Unknown')
            except:
                return 'Unknown'

        df['continent'] = df['country'].apply(get_continent)
        return df
    
    @staticmethod
    def normalize_country_names(df: pd.DataFrame):
        def normalize_name(name):
            try:
                country = pycountry.countries.get(name=name)
                if not country:
                    country = pycountry.countries.search_fuzzy(name)[0]
                return country.name
            except:
                return name

        df['country'] = df['country'].apply(normalize_name)
        return df