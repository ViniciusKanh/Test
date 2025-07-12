import json
from marca.interest_measures import InterestMeasuresGroup
groups = json.load(open('presets/interest_measures_groups/_interest_measures_groups.json', 'r'))

def load_interest_measures_group(group_name):
    return InterestMeasuresGroup(group_name, groups[group_name])