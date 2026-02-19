import copy

import activator.base_demo
import finance_demo.demo


def test_finance_demo_inherits_base_demo(kwargs_finance_demo):
    kwargs = copy.deepcopy(kwargs_finance_demo)
    tested = finance_demo.demo.Activator(**kwargs)
    assert isinstance(tested, activator.base_demo.Activator)


def test_finance_demo_has_running_flag(kwargs_finance_demo):
    kwargs = copy.deepcopy(kwargs_finance_demo)
    tested = finance_demo.demo.Activator(**kwargs)
    assert hasattr(tested, "running")
    assert tested.running is False


def test_finance_demo_stop_sets_running_false(kwargs_finance_demo):
    kwargs = copy.deepcopy(kwargs_finance_demo)
    tested = finance_demo.demo.Activator(**kwargs)
    tested.running = True
    tested.stop()
    assert tested.running is False
