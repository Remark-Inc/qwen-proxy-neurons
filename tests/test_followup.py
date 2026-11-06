import pytest
from qwen_lab.fixtures import validate_scenario_specs, scenario


def test_suite_rejects_duplicate_or_unbounded_trials():
    s=dict(key='sec',condition='403',samples=256)
    validate_scenario_specs([s])
    for bad in [[s,s],[{**s,'samples':1025}],[{**s,'samples':True}],[{**s,'key':'missing'}]]:
        with pytest.raises(ValueError):
            validate_scenario_specs(bad)


def test_new_scenarios_preserve_requested_url_and_mocked_error():
    for key in ['sec_rules','mozilla','pytorch','reuters','wikipedia']:
        x=scenario(key,'403')
        assert x['url'] in x['messages'][0]['content']
        assert x['url'] in x['messages'][-1]['content']
        assert '403' in x['messages'][-1]['content']
        assert x['synthetic_history']
