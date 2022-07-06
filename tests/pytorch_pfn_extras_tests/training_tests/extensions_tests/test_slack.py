from unittest import mock
import requests

import pytest

import pytorch_pfn_extras as ppe


@pytest.mark.skipif(
    not ppe.training.extensions.slack._slack_sdk_available,
    reason="Slack SDK not installed"
)
class TestSlack:
    def _get_manager(self):
        return ppe.training.ExtensionsManager({}, [], 100, iters_per_epoch=5)

    @pytest.mark.parametrize('use_threads',[False, True])
    def test_post_message(self, use_threads):
        manager = self._get_manager()
        message = 'It {.iteration} loss: {loss}'
        extension = ppe.training.extensions.Slack(
            '0', message, token='123', use_threads=use_threads,
            trigger=(1, 'iteration'))

        t_ts = None
        if use_threads:
            t_ts = 1

        manager.extend(extension, trigger=(1, 'iteration'))
        with mock.patch(
            'slack_sdk.WebClient.chat_postMessage',
            return_value={'ok': True, 'ts': t_ts},
        ) as patched:
            with manager.run_iteration():
                ppe.reporting.report({'loss': 0.5})
            patched.assert_called_with(
                channel='0', text='It 1 loss: 0.5', thread_ts=None
            )
            with manager.run_iteration():
                ppe.reporting.report({'loss': 0.75})
            patched.assert_called_with(
                channel='0', text='It 2 loss: 0.75', thread_ts=t_ts
            )

    def test_post_message_webhook(self):
        manager = self._get_manager()
        message = 'It {.iteration} loss: {loss}'
        extension = ppe.training.extensions.Slack(
            '0', message, webhook_url="http://test", trigger=(1, 'iteration'))

        manager.extend(extension, trigger=(1, 'iteration'))
        response = requests.Response()
        response.status_code = 200
        with mock.patch(
            'requests.post',
            return_value=response,
        ) as patched:
            with manager.run_iteration():
                ppe.reporting.report({'loss': 0.5})
            patched.assert_called_with(
                'http://test', '{"text": "It 1 loss: 0.5"}'
            )
            with manager.run_iteration():
                ppe.reporting.report({'loss': 0.75})
            patched.assert_called_with(
                'http://test', '{"text": "It 2 loss: 0.75"}'
            )

    @pytest.mark.parametrize(
        'message',
        [
            'It {.iteration} loss: {loss} custom: {.foo}',
            lambda m,c,o: 'It {.iteration} loss: {loss} custom: {.foo}'.format(m,c, **o)
        ]
    )
    def test_post_message_context(self, message):
        class _CustomContext:
            def __init__(self):
                self.foo = 'bar'

        manager = self._get_manager()
        context = _CustomContext()
        extension = ppe.training.extensions.Slack(
            '0', message, context_object=context, token='123', trigger=(1, 'iteration'))
        manager.extend(extension, trigger=(1, 'iteration'))
        with mock.patch(
            'slack_sdk.WebClient.chat_postMessage',
            return_value={'ok': True, 'ts': 1},
        ) as patched:
            with manager.run_iteration():
                ppe.reporting.report({'loss': 0.5})
            patched.assert_called_with(
                channel='0', text='It 1 loss: 0.5 custom: bar',
                thread_ts=None
            )
            context.foo = 'test'
            with manager.run_iteration():
                ppe.reporting.report({'loss': 0.75})
            patched.assert_called_with(
                channel='0', text='It 2 loss: 0.75 custom: test',
                thread_ts=None
            )

    def test_post_message_files(self):
        manager = self._get_manager()
        message = 'it: {.iteration}'
        filenames = ['file_{.iteration}', '{._out}/abc']
        extension = ppe.training.extensions.Slack(
            '0', message, filenames_template=filenames, token='123', trigger=(1, 'iteration'))
        manager.extend(extension, trigger=(1, 'iteration'))

        with mock.patch(
            'slack_sdk.WebClient.chat_postMessage',
            return_value={'ok': True, 'ts': 1},
        ), mock.patch('slack_sdk.WebClient.files_upload') as upload:
            with manager.run_iteration():
                pass
            upload.assert_has_calls([
                mock.call(channels=r'0', file='file_1', thread_ts=None),
                mock.call(channels=r'0', file='result/abc', thread_ts=None),
            ], any_order=True)

    def test_invalid_combinations(self):
        message = 'it: {.iteration}'
        filenames = ['file_{.iteration}', '{._out}/abc']
        with pytest.raises(ValueError, match='used to post files'):
            ppe.training.extensions.Slack(
                '0', message, None, None, filenames, webhook_url='123', trigger=(1, 'iteration'))
        with pytest.raises(ValueError, match='client and token'):
            ppe.training.extensions.Slack(
                '0', message, None, None, filenames, webhook_url='123',
                client=1, trigger=(1, 'iteration'))
        with pytest.raises(ValueError, match='client and token'):
            ppe.training.extensions.Slack(
                '0', message, None, None, filenames, webhook_url='123',
                token=1, trigger=(1, 'iteration'))
        with pytest.raises(ValueError, match='client and token'):
            ppe.training.extensions.Slack(
                '0', message, None, None, filenames, webhook_url='123',
                client=1, token=1, trigger=(1, 'iteration'))
