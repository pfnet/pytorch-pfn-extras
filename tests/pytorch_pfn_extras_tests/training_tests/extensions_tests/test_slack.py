import json
from unittest import mock

import pytest

import pytorch_pfn_extras as ppe


@pytest.mark.skipif(
    not ppe.training.extensions.slack._slack_sdk_available,
    reason="Slack SDK not installed"
)
class TestSlack:
    def _get_manager(self):
        return ppe.training.ExtensionsManager({}, [], 1, iters_per_epoch=5)

    @pytest.mark.parametrize('thread',[False, True])
    def test_post_message(self, thread):
        manager = self._get_manager()
        message = 'It {manager.iteration} loss: {loss}'
        extension = ppe.training.extensions.Slack(
            '0', message, token='123', thread=thread)

        t_ts = None
        if thread:
            t_ts = 1

        manager.extend(extension, trigger=(1, 'iteration'))
        with mock.patch(
            'slack_sdk.WebClient.chat_postMessage',
            return_value={'ok': True, 'ts': t_ts},
        ) as patched:
            with manager.run_iteration():
                assert 'Training started' in patched.call_args.kwargs["text"]
                ppe.reporting.report({'loss': 0.5})
            patched.assert_called_with(
                channel='0', text='It 1 loss: 0.5', thread_ts=t_ts
            )
            with manager.run_iteration():
                ppe.reporting.report({'loss': 0.75})
            patched.assert_called_with(
                channel='0', text='It 2 loss: 0.75', thread_ts=t_ts
            )
            with manager.run_iteration():
                ppe.reporting.report({'loss': 0.75})
            with manager.run_iteration():
                ppe.reporting.report({'loss': 0.75})
            with manager.run_iteration():
                ppe.reporting.report({'loss': 0.75})
            assert 'Training finish' in patched.call_args.kwargs["text"]

    def test_post_message_on_error(self):
        manager = self._get_manager()
        message = 'It {manager.iteration} loss: {loss}'
        extension = ppe.training.extensions.Slack(
            '0', message, token='123', thread=False)

        t_ts = None

        manager.extend(extension, trigger=(1, 'iteration'))
        with mock.patch(
            'slack_sdk.WebClient.chat_postMessage',
            return_value={'ok': True, 'ts': t_ts},
        ) as patched:
            try:
                with manager.run_iteration():
                    raise RuntimeError('error')
            except RuntimeError:
                assert 'Error during' in patched.call_args.kwargs["text"]

    def test_post_message_webhook(self):
        manager = self._get_manager()
        message = 'It {manager.iteration} loss: {loss}'
        extension = ppe.training.extensions.SlackWebhook(
            webhook_url="http://test", msg=message)

        manager.extend(extension, trigger=(1, 'iteration'))
        payload_1 = json.dumps({'text': "It 1 loss: 0.5"}).encode('utf-8')
        payload_2 = json.dumps({'text': "It 2 loss: 0.75"}).encode('utf-8')
        with mock.patch(
            'urllib.request.urlopen'
        ) as patched:
            with manager.run_iteration():
                ppe.reporting.report({'loss': 0.5})
            assert patched.call_args.args[0].data == payload_1
            with manager.run_iteration():
                ppe.reporting.report({'loss': 0.75})
            assert patched.call_args.args[0].data == payload_2

    @pytest.mark.parametrize(
        'message',
        [
            'It {manager.iteration} loss: {loss} custom: {context.foo}',
            lambda m,c,o: 'It {manager.iteration} loss: {loss} custom: {context.foo}'.format(  # NOQA
                manager=m, context=c, **o)
        ]
    )
    def test_post_message_context(self, message):
        class _CustomContext:
            def __init__(self):
                self.foo = 'bar'

        manager = self._get_manager()
        context = _CustomContext()
        extension = ppe.training.extensions.Slack(
            '0', message, context_object=context, token='123')
        manager.extend(extension, trigger=(1, 'iteration'))
        with mock.patch(
            'slack_sdk.WebClient.chat_postMessage',
            return_value={'ok': True, 'ts': 1},
        ) as patched:
            with manager.run_iteration():
                ppe.reporting.report({'loss': 0.5})
            patched.assert_called_with(
                channel='0', text='It 1 loss: 0.5 custom: bar',
                thread_ts=1
            )
            context.foo = 'test'
            with manager.run_iteration():
                ppe.reporting.report({'loss': 0.75})
            patched.assert_called_with(
                channel='0', text='It 2 loss: 0.75 custom: test',
                thread_ts=1
            )

    def test_post_message_files(self):
        manager = self._get_manager()
        message = 'it: {manager.iteration}'
        filenames = ['file_{manager.iteration}', '{manager._out}/abc']
        extension = ppe.training.extensions.Slack(
            '0', message, filenames=filenames, token='123')
        manager.extend(extension, trigger=(1, 'iteration'))

        with mock.patch(
            'slack_sdk.WebClient.chat_postMessage',
            return_value={'ok': True, 'ts': 1},
        ), mock.patch('slack_sdk.WebClient.files_upload') as upload:
            with manager.run_iteration():
                pass
            upload.assert_has_calls([
                mock.call(file='file_1'),
                mock.call(file='result/abc'),
            ], any_order=True)

    def test_invalid(self):
        message = 'it: {manager.iteration}'
        filenames = ['file_{manager.iteration}', '{manager._out}/abc']
        with pytest.raises(RuntimeError, match='needed for communicating'):
            ppe.training.extensions.Slack(
                '0', message, start_msg=None, end_msg=None, filenames=filenames)
