from pytorch_pfn_extras.training.extensions import log_report


def test_log_buffer():
    buf = log_report._LogBuffer()
    looker = buf.emit_new_looker()
    assert buf.size() == 0
    buf.append('mes1')
    buf.append('mes2')
    assert buf.size() == 2
    assert looker.get() == ['mes1', 'mes2']
    assert buf.size() == 2
    looker.clear()
    assert buf.size() == 0
    assert looker.get() == []
    buf.append('mes3')
    assert buf.size() == 1
    assert looker.get() == ['mes3']
    assert buf.size() == 1
    looker.clear()
    assert buf.size() == 0
    assert looker.get() == []


def test_log_buffer_multiple_lookers():
    buf = log_report._LogBuffer()
    looker1 = buf.emit_new_looker()
    looker2 = buf.emit_new_looker()
    buf.append('mes1')
    assert looker1.get() == ['mes1']
    assert looker2.get() == ['mes1']
    assert buf.size() == 1
    looker2.clear()
    assert buf.size() == 1
    buf.append('mes2')
    assert looker1.get() == ['mes1', 'mes2']
    assert looker2.get() == ['mes2']
    assert buf.size() == 2
    looker2.clear()
    assert buf.size() == 2
    looker1.clear()
    assert buf.size() == 0
