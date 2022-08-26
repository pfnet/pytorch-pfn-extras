import getpass
import os
import json
import urllib.request
import shlex
import sys
import socket
import traceback
import types
from typing import Any, Callable, Optional, Sequence, Union
import warnings


from pytorch_pfn_extras.training import extension
from pytorch_pfn_extras.training import trigger as trigger_module
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol
from pytorch_pfn_extras.training._trigger_util import TriggerLike


try:
    import slack_sdk
    _slack_sdk_available = True
except ImportError:
    _slack_sdk_available = False


def _failsafe(func: Callable[[], Any]) -> str:
    # getpass.getuser() fails when user account does not exist.
    try:
        return str(func())
    except Exception:
        return 'UNKNOWN'


_identity = (
    f'{_failsafe(getpass.getuser)}@{_failsafe(socket.gethostname)} '
    f'[PID {_failsafe(os.getpid)}]')


def _default_msg(
        manager: ExtensionsManagerProtocol,
        context: Any,
) -> str:
    return f'Epoch #{manager.epoch}'


def _default_start_msg(
        manager: ExtensionsManagerProtocol,
        context: Any,
) -> str:
    cmdline = ' '.join([shlex.quote(x) for x in sys.argv])
    return (
        f'🏃 *Training started! {_identity}*\n'
        f'Command: `{cmdline}`'
    )


def _default_end_msg(
        manager: ExtensionsManagerProtocol,
        context: Any,
) -> str:
    return f'✅ *Training finished! {_identity}*'


def _default_error_msg(
        manager: ExtensionsManagerProtocol,
        exc: Exception,
        context: Any,
) -> str:
    return (
        f'❌ *Error during training. {_identity}*\n'
        f'{type(exc).__name__}: {exc}\n'
        'Traceback:\n'
        '```\n'
        ''.join(traceback.format_tb(exc.__traceback__)).strip() + '\n'
        '```'
    )


_MessageFunc = Callable[[ExtensionsManagerProtocol, Any], str]
_ErrorMessageFunc = Callable[[ExtensionsManagerProtocol, Any, Exception], str]
_FilenamesFunc = Callable[[ExtensionsManagerProtocol, Any], Sequence[str]]

_message_spec_doc = """
    This extension posts a message when:

    * ``start_msg``: The training has started
    * ``msg``: The extension is triggered, usually at the end of each epoch
    * ``end_msg``: The training has finished
    * ``error_msg``: An exception has raised during the training

    These messages can be specified as a format string, a callable that
    returns a string, or None to disable posting on that event.

    When using a format string, the following variables are available for use:

    * ``manager``: an ExtensionsManager object
    * ``default``: the default message string
    * ``context``: an arbitrary object passed to this extension
    * ``error``: an Exception object (for ``error_msg`` only)
    * All reported values (``manager.observations``)

    When using a callable, it should take `(ExtensionsManager, context)` or
    `(ExtensionsManager, Exception, context)` (for ``error_msg``) and return
    a string.
"""


class _SlackBase(extension.Extension):

    trigger: TriggerLike = (1, 'epoch')

    default_msg = _default_msg
    default_start_msg = _default_start_msg
    default_end_msg = _default_end_msg
    default_error_msg = _default_error_msg

    def __init__(self) -> None:
        self._available = True
        self._msg: Optional[Union[_MessageFunc, str]] = None
        self._start_msg: Optional[Union[_MessageFunc, str]] = None
        self._end_msg: Optional[Union[_MessageFunc, str]] = None
        self._error_msg: Optional[Union[_ErrorMessageFunc, str]] = None
        self._context: Any = None
        self._filenames: Optional[Union[_FilenamesFunc, Sequence[str]]] = None
        self._upload_trigger: Optional[trigger_module.Trigger] = None

    def _post_message(self, text: str) -> None:
        raise NotImplementedError

    def _upload_files(self, filenames: Sequence[str]) -> Sequence[str]:
        raise NotImplementedError

    def _format(
            self,
            msg: Union[_MessageFunc, str],
            default: Optional[_MessageFunc],
            manager: ExtensionsManagerProtocol,
    ) -> str:
        default_str = '' if default is None else default(manager, self._context)
        if isinstance(msg, str):
            return msg.format(
                manager=manager,
                context=self._context,
                default=default_str,
                **manager.observation
            )
        return msg(manager, self._context)

    def _format_error(
            self,
            manager: ExtensionsManagerProtocol,
            error: Exception,
    ) -> str:
        msg = self._error_msg
        assert msg is not None
        if isinstance(msg, str):
            return msg.format(
                manager=manager,
                context=self._context,
                default=_default_error_msg(manager, error, self._context),
                error=error
            )
        return msg(manager, error, self._context)

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        if not self._available:
            return

        filenames: Sequence[str] = []
        if self._filenames is None:
            pass
        elif isinstance(self._filenames, Sequence):
            filenames = [
                self._format(f, None, manager) for f in self._filenames]
        else:  # callable
            filenames = self._filenames(manager, self._context)

        needs_upload = (
            len(filenames) != 0
            and (self._upload_trigger is None
                 or self._upload_trigger(manager)))

        if self._msg is None and not needs_upload:
            # The message is not set and no files to upload.
            return

        text = ''
        if self._msg is not None:
            text = self._format(self._msg, _default_msg, manager)

        # TODO(kmaehashi): keep track of already uploaded files and warn
        # TODO(kmaehashi): warn too many or too large files

        attachments = ''
        if needs_upload:
            permalinks = self._upload_files(filenames)
            attachments = ''.join([f'<{link}| >' for link in permalinks])

        self._post_message(text + attachments)

    def initialize(self, manager: ExtensionsManagerProtocol) -> None:
        if not self._available or self._start_msg is None:
            return
        self._post_message(
            self._format(self._start_msg, _default_start_msg, manager))

    def finalize(self, manager: ExtensionsManagerProtocol) -> None:
        if not self._available or self._end_msg is None:
            return
        self._post_message(
            self._format(self._end_msg, _default_end_msg, manager))

    def on_error(
            self,
            manager: ExtensionsManagerProtocol,
            exc: Exception,
            tb: types.TracebackType
    ) -> None:
        if not self._available or self._error_msg is None:
            return
        self._post_message(self._format_error(manager, exc))


class Slack(_SlackBase):
    __doc__ = """An extension to communicate with Slack.

    .. admonition:: Example

        >>> ppe.training.extensions.Slack(
        ...     channel="experiment-progress",
        ...     msg="Epoch #{manager.epoch}: loss = {val/loss}",
        ...     end_msg="{default} \\n <@username> Check out the result!",
        ...
        ...     # Upload files at the end of the experiment.
        ...     filenames=["result/statistics.png"],
        ...     upload_trigger=(max_epoch, 'epoch'),
        ... )
    """ + _message_spec_doc + """
    This extension can upload files along with the message when triggered.
    ``filenames`` can be a list of filenames (the same formatting rule as
    ``msg`` apply), or a callable taking (ExtensionsManager, context) and
    returning a list of filenames.

    To use this extension, you must create a Slack app, then specify the
    token via an environment variable ``SLACK_BOT_TOKEN`` or ``token``
    option.

    Args:
        channel (str): The channel where messages and files will be sent.
            This can be a channel name or a channel ID.
        msg (str, callable, or None): A message to be sent when triggered.
            It can be a string to be formatted using ``.format`` or a callable
            that returns a string.
        start_msg (str, callable, or None): A message to be sent
            at the beginning of the experiment.
        end_msg (str, callable, or None): A message to be sent
            at the completion of the experiment.
        error_msg (str, callable, or None): A message to be sent
            when an exception is raised during the experiment.
        thread (bool): When True, subsequent messages will be
            posted as a thread of the original message.
            Default is ``True``.
        filenames (list of str or callable): A list of files that will
            be uploaded. These are string templates that can take
            values in the same way as ``msg``, or a callable that returns a
            list of filenames.
        upload_trigger (trigger or None): Used to upload files at certain events.
            If not specified, files will be uploaded in every call.
        context: Any arbitrary user object you will need when
            generating a message.
        token (str): Slack bot token. If ``None``, the environment
            variable ``SLACK_BOT_TOKEN`` will be used.
            Optional, default is ``None``.
    """

    trigger: TriggerLike = (1, 'epoch')

    def __init__(
        self,
        channel: str,
        msg: Optional[Union[str, _MessageFunc]] = None,
        *,
        start_msg: Optional[Union[str, _MessageFunc]] = '{default}',
        end_msg: Optional[Union[str, _MessageFunc]] = '{default}',
        error_msg: Optional[Union[str, _ErrorMessageFunc]] = '{default}',
        thread: bool = True,
        filenames: Optional[Union[Sequence[str], _FilenamesFunc]] = None,
        upload_trigger: Optional[TriggerLike] = None,
        context: Any = None,
        token: Optional[str] = None,
    ) -> None:
        super().__init__()
        if not _slack_sdk_available:
            self._available = False
            warnings.warn(
                '`slack_sdk` package is unavailable. '
                'The Slack extension will do nothing.')
            return

        self._channel = channel
        self._msg = msg
        self._start_msg = start_msg
        self._end_msg = end_msg
        self._error_msg = error_msg
        self._context = context
        self._thread = thread
        self._filenames = filenames
        self._upload_trigger = None
        if upload_trigger is not None:
            self._upload_trigger = trigger_module.get_trigger(upload_trigger)

        if token is None:
            token = os.environ.get('SLACK_BOT_TOKEN', None)
        if token is None:
            raise RuntimeError(
                'A bot `token` is needed for communicating with Slack')
        self._client = slack_sdk.WebClient(token=token)
        self._thread_ts: Optional[str] = None

    def _upload_files(self, filenames: Sequence[str]) -> Sequence[str]:
        permalinks = []
        try:
            for filename in filenames:
                response = self._client.files_upload(file=filename)
                assert response.get("ok")  # type: ignore[no-untyped-call]
                permalinks.append(response['file']['permalink'])
        except Exception as e:
            warnings.warn(
                f'Slack upload failed: {type(e).__name__}: {e} '
                f'[{filenames}]')
        return permalinks

    def _post_message(self, text: str) -> None:
        try:
            response = self._client.chat_postMessage(
                channel=self._channel,
                text=text,
                thread_ts=self._thread_ts,
            )
            assert response.get("ok")  # type: ignore[no-untyped-call]
            if self._thread and self._thread_ts is None:
                ts = response.get("ts")  # type: ignore[no-untyped-call]
                self._thread_ts = ts
        except Exception as e:
            warnings.warn(
                f'Slack post failed: {type(e).__name__}: {e} '
                f'[{text}]')


class SlackWebhook(_SlackBase):
    __doc__ = """An extension to communicate with Slack using Incoming Webhook.

    .. admonition:: Example

        >>> ppe.training.extensions.SlackWebhook(
        ...     url="https://hooks.slack.com/services/Txxxxx.....",
        ...     msg="Epoch #{manager.epoch}: loss = {val/loss}",
        ...     end_msg="{default} \\n <@username> Check out the result!",
        ... )
    """ + _message_spec_doc + """
    Args:
        url (str): Incoming webhook URL to send messages.
        msg (str, callable, or None): A message to be sent when triggered.
            It can be a string to be formatted using ``.format`` or a callable
            that returns a string.
        start_msg (str, callable, or None): A message to be sent
            at the beginning of the experiment.
        end_msg (str, callable, or None): A message to be sent
            at the completion of the experiment.
        error_msg (str, callable, or None): A message to be sent
            when an exception is raised during the experiment.
        context (object): Any arbitrary user object you will need when
            generating a message.
    """

    def __init__(
        self,
        url: str,
        msg: Optional[Union[str, _MessageFunc]] = None,
        *,
        start_msg: Optional[Union[str, _MessageFunc]] = '{default}',
        end_msg: Optional[Union[str, _MessageFunc]] = '{default}',
        error_msg: Optional[Union[str, _ErrorMessageFunc]] = '{default}',
        context: Any = None,
    ) -> None:
        super().__init__()
        self._url = url
        self._msg = msg
        self._start_msg = start_msg
        self._end_msg = end_msg
        self._error_msg = error_msg
        self._context = context

    def _post_message(self, text: str) -> None:
        payload = json.dumps({'text': text}).encode('utf-8')
        request_headers = {'Content-Type': 'application/json; charset=utf-8'}
        request = urllib.request.Request(
            url=self._url,
            data=payload,
            method='POST',
            headers=request_headers,
        )
        try:
            response = urllib.request.urlopen(request)
            assert 200 <= response.status < 300, response
        except Exception as e:
            warnings.warn(
                f'Slack WebHook request failed: {type(e).__name__}: {e} '
                f'[{text}]')
