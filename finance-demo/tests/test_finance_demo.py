import activator.base_demo
import finance_demo.demo


def test_finance_demo_inherits_base_demo(kwargs_finance_demo):
    demo = finance_demo.demo.Activator(**kwargs_finance_demo)
    assert isinstance(demo, activator.base_demo.Activator)


def test_finance_demo_has_running_flag(kwargs_finance_demo):
    demo = finance_demo.demo.Activator(**kwargs_finance_demo)
    assert hasattr(demo, "running")
    assert demo.running is False


def test_finance_demo_stop_sets_running_false(kwargs_finance_demo):
    demo = finance_demo.demo.Activator(**kwargs_finance_demo)
    demo.running = True
    demo.stop()
    assert demo.running is False
