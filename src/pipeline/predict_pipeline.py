import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            features_transformed = preprocessor.transform(features)
            predictions = model.predict(features_transformed)
            return predictions

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        funding_total_usd,
        funding_rounds,
        avg_participants,
        relationships,
        milestones,
        has_angel,
        has_VC,
        has_roundA,
        has_roundB,
        has_roundC,
        has_roundD,
        is_CA,
        is_NY,
        is_MA,
        is_TX,
        is_otherstate,
        is_software,
        is_web,
        is_mobile,
        is_enterprise,
        is_advertising,
        is_gamesvideo,
        is_ecommerce,
        is_consulting,
        is_othercategory,
        is_top500,
        age_first_funding_year,
        age_last_funding_year,
        age_first_milestone_year,
        age_last_milestone_year,
        state_code,
        category_code
    ):
        self.funding_total_usd = funding_total_usd
        self.funding_rounds = funding_rounds
        self.avg_participants = avg_participants
        self.relationships = relationships
        self.milestones = milestones
        self.has_angel = has_angel
        self.has_VC = has_VC
        self.has_roundA = has_roundA
        self.has_roundB = has_roundB
        self.has_roundC = has_roundC
        self.has_roundD = has_roundD
        self.is_CA = is_CA
        self.is_NY = is_NY
        self.is_MA = is_MA
        self.is_TX = is_TX
        self.is_otherstate = is_otherstate
        self.is_software = is_software
        self.is_web = is_web
        self.is_mobile = is_mobile
        self.is_enterprise = is_enterprise
        self.is_advertising = is_advertising
        self.is_gamesvideo = is_gamesvideo
        self.is_ecommerce = is_ecommerce
        self.is_consulting = is_consulting
        self.is_othercategory = is_othercategory
        self.is_top500 = is_top500
        self.age_first_funding_year = age_first_funding_year
        self.age_last_funding_year = age_last_funding_year
        self.age_first_milestone_year = age_first_milestone_year
        self.age_last_milestone_year = age_last_milestone_year
        self.state_code = state_code
        self.category_code = category_code

    def get_data_as_data_frame(self):
        try:
            data = {
                "funding_total_usd": [self.funding_total_usd],
                "funding_rounds": [self.funding_rounds],
                "avg_participants": [self.avg_participants],
                "relationships": [self.relationships],
                "milestones": [self.milestones],
                "has_angel": [self.has_angel],
                "has_VC": [self.has_VC],
                "has_roundA": [self.has_roundA],
                "has_roundB": [self.has_roundB],
                "has_roundC": [self.has_roundC],
                "has_roundD": [self.has_roundD],
                "is_CA": [self.is_CA],
                "is_NY": [self.is_NY],
                "is_MA": [self.is_MA],
                "is_TX": [self.is_TX],
                "is_otherstate": [self.is_otherstate],
                "is_software": [self.is_software],
                "is_web": [self.is_web],
                "is_mobile": [self.is_mobile],
                "is_enterprise": [self.is_enterprise],
                "is_advertising": [self.is_advertising],
                "is_gamesvideo": [self.is_gamesvideo],
                "is_ecommerce": [self.is_ecommerce],
                "is_consulting": [self.is_consulting],
                "is_othercategory": [self.is_othercategory],
                "is_top500": [self.is_top500],
                "age_first_funding_year": [self.age_first_funding_year],
                "age_last_funding_year": [self.age_last_funding_year],
                "age_first_milestone_year": [self.age_first_milestone_year],
                "age_last_milestone_year": [self.age_last_milestone_year],
                "state_code": [self.state_code],
                "category_code": [self.category_code],
            }

            return pd.DataFrame(data)

        except Exception as e:
            raise CustomException(e, sys)
