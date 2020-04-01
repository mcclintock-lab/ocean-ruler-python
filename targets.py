
import constants

TARGET_FILES = {constants.LOBSTER:"lobster_target", 
                constants.SCALLOP:"scallop_target", 
                constants.ABALONE:"abalone_target",
                constants.FINFISH: "finfish_target"}


def get_target_file(fishery_type):
    for target, target_file in TARGET_FILES.items():
        if target in fishery_type:
            return target_file
    
    return None