import getpass
import os
import json
import urllib.request
import shlex
import sys
import socket
import traceback
import types
import warnings

from typing import Any, Callable, Dict, List, Optional, Union
from typing import TYPE_CHECKING, cast

from pytorch_pfn_extras.training import extension
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol


if TYPE_CHECKING:
    from pytorch_pfn_extras.training._trigger_util import TriggerLike


try:
    import slack_sdk
    _slack_sdk_available = True
except ImportError:
    _slack_sdk_available = False


_identity = f'{getpass.getuser()}@{socket.gethostname()} [PID {os.getpid()}]'


def _default_start_msg(m: ExtensionsManagerProtocol, c: Any, o: Dict[Any, Any]) -> str:
    return f'''🏃 *Training started! {_identity}*
Command: `{' '.join([shlex.quote(x) for x in sys.argv])}`
'''


_default_end_msg = f'✅ *Training finished! {_identity}*'


def _default_error_msg(
    m: ExtensionsManagerProtocol,
    c: Any,
    o: Dict[Any, Any],
    exc: Exception
) -> str:
    return f'''❌ *Error during training. {_identity}*
{type(exc).__name__}: {exc}
Traceback:
```
{''.join(traceback.format_tb(exc.__traceback__)).strip()}
```'''


class Slack(extension.Extension):
    """An extension to communicate with Slack.

    This extension receives a ``msg`` argument that contains
    placeholders that will be populated with ``manager`` or ``manager.observation``
    fields, and any arbitrary user-defined object passed to ``context``.

    For example. `msg = "Loss {val/loss} for iteration {manager.iteration}"`
    will be populated as `msg.format(manager, context_object, **observation)`
    retrieving the ``loss`` value from the ``observation`` and ``.iteration`` from
    the manager object. Instead of string, a callable object taking the
    ``ExtensionsManager``, ``context_object`` and the observation dictionary can
    be used instead. The same rules apply for the ''start_msg``, ``end_msg`` and
    ``filenames_template`` arguments. ``error_msg`` also takes the associated
    ``Exception`` object as an argument when passed as a callable.

    Args:
        channel (str): The channel where messages or files will be sent.
            This can be the channel name if starts with '#' or the channel id
            otherwise.
        msg (str or callable): Template for sending the message.
            It can be a string to be formatted using ``.format`` or a callable
            that returns a string. In both cases, `manager`, the current
            observation dict and the context object are used to look up
            values of interest.
        start_msg (str or callable): Template for sending a message
            at the beggining of the experiment. The default
            start message will be sent if not specified.
            To avoid sending a message use ``None``.
            See ``msg`` for format.
        end_msg (str or callable): Template for sending a message
            at the completion of the experiment. The default
            end message will be sent if not specified.
            To avoid sending a message use ``None``.
            See ``msg`` for format.
        error_msg (str or callable): Template for sending a message
            when an error is detected. The default
            error message will be sent if not specified.
            To avoid sending a message use ``None``.
            See ``msg`` for format.
        filenames (list of str or callable): list of files that will
            be uploaded to slack, these are string templates that can take
            values in the same way as ``msg``. Optional.
        thread (bool): If subsequent calls of this extension should be
            posted as a thread of the original message or not.
            Default is ``True``.
        context (object): Any arbitrary object you need to
            format the text or the filenames. Optional, default is ``None``.
        token (str): Token for the slack api, if ``None`` the environment
            variable ``SLACK_TOKEN`` will be used. Ignored if ``client`` is
            supplied. Optional, default is ``None``.
        client (slack_sdk.WebClient): In case that there is an already created
            slack client in the application, allows to directly use it.
            Optional, default is ``None``
    """

    trigger: 'TriggerLike' = (1, 'epoch')

    def __init__(
        self,
        channel: str,
        msg: Union[
            str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
        ],
        *,
        start_msg: Optional[Union[
            str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
        ]] = _default_start_msg,
        end_msg: Optional[Union[
            str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
        ]] = _default_end_msg,
        error_msg: Optional[Union[
            str, Callable[[ExtensionsManagerProtocol, Any, dict, Exception], str]
        ]] = _default_error_msg,
        filenames: Optional[
            List[
                Union[
                    str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
                ]
            ]
        ] = None,
        thread: bool = True,
        context_object: Optional[object] = None,
        token: Optional[str] = None,
        client: Optional[Any] = None,  # slack_sdk.WebClient, Any to avoid mypy errors
    ) -> None:
        if not _slack_sdk_available:
            warnings.warn(
                '`slack_sdk` package is unavailable. '
                'The Slack extension will do nothing.')
            return

        self._client = client
        if client is None:
            if token is None:
                token = os.environ.get('SLACK_TOKEN', None)
            if token is None:
                raise RuntimeError(
                    '`token` is needed for communicating with slack')
            self._client = slack_sdk.WebClient(token=token)

        # values in current observation or log report to send to slack
        self._msg = msg
        self._start_msg = start_msg  # type: ignore[assignment]
        self._end_msg = end_msg  # type: ignore[assignment]
        self._error_msg = error_msg  # type: ignore[assignment]
        if filenames is None:
            filenames = []
        self._filenames = filenames
        self._context = context_object
        self._channel_id = self._get_channel_id(channel)
        self._thread = thread
        self._ts = None

    def _get_channel_id(self, channel: str) -> str:
        if channel[0] != '#':
            return channel
        channel_ids = {}
        assert self._client is not None
        # TODO enable pagination
        response = self._client.conversations_list()
        for c in response['channels']:
            channel_ids[c['name']] = c['id']
        channel_name = channel_ids.get(channel[1:], None)
        if channel_name is None:
            raise RuntimeError(f'Couldn\'t find channel {channel}')
        return cast(str, channel_name)

    def _send_message(
        self,
        manager: ExtensionsManagerProtocol,
        text: Union[
            str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
        ],
        error: Optional[Exception] = None
    ) -> None:
        if not _slack_sdk_available:
            return

        observation = manager.observation
        if callable(text):
            if error is None:
                text = text(manager, self._context, observation)
            else:
                text = text(  # type: ignore[call-arg]
                    manager, self._context, observation, error)
        else:
            text = text.format(
                manager=manager, context=self._context, **observation)

        assert self._client is not None

        ts = None
        if self._thread:
            ts = self._ts

        for filename in self._filenames:
            if callable(filename):
                filename = filename(manager, self._context, observation)
            else:
                filename = filename.format(
                    manager=manager, context=self._context, **observation)
            response = self._client.files_upload(
                file=filename,
                thread_ts=ts,
            )
            text += '<' + response['file']['permalink'] + '| >'
            assert response.get("ok")  # type: ignore[no-untyped-call]

        response = self._client.chat_postMessage(
            channel=self._channel_id,
            text=text,
            thread_ts=ts,
        )
        assert response.get("ok")  # type: ignore[no-untyped-call]
        if self._thread and ts is None:
            ts = response.get("ts")  # type: ignore[no-untyped-call]
            self._ts = ts

    def initialize(self, manager: ExtensionsManagerProtocol) -> None:
        if _slack_sdk_available and self._start_msg is not None:
            self._send_message(manager, self._start_msg)

    def finalize(self, manager: ExtensionsManagerProtocol) -> None:
        if _slack_sdk_available and self._end_msg is not None:
            self._send_message(manager, self._end_msg)

    def on_error(
            self,
            manager: ExtensionsManagerProtocol,
            exc: Exception,
            tb: types.TracebackType
    ) -> None:
        if _slack_sdk_available and self._error_msg is not None:
            self._send_message(
                manager, self._error_msg, error=exc)  # type: ignore[call-arg,arg-type]

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        if _slack_sdk_available:
            self._send_message(manager, self._msg)


class SlackWebhook(extension.Extension):
    """An extension to communicate with Slack.

    This extension receives a ``msg`` argument that contains
    placeholders that will be populated with ``manager`` or ``manager.observation``
    fields, and custom objects such as ``context_object``.

    For example. `msg = "Loss {val/loss} for iteration {manager.iteration}"`
    will be populated as `msg.format(manager, context_object, **observation)`
    retrieving the ``loss`` value from the ``observation`` and ``.iteration`` from
    the manager object. Instead of string, a callable object taking the
    ``ExtensionsManager``, ``context_object`` and the observation dictionary can
    be used instead. The same rules apply for the ''start_msg``, ``end_msg`` and
    ``filenames_template`` arguments. ``error_msg`` also takes the associated
    ``Exception`` object as an argument when passed as a callable.

    Args:
        webhook_url (str): Incoming webhook URL to send messages.
        msg (str or callable): Template for sending the message.
            It can be a string to be formatted using ``.format`` or a callable
            that returns a string. In both cases, `manager`, the current
            observation dict and the context object are used to look up
            values of interest.
        start_msg (str or callable): Template for sending a message
            at the beggining of the experiment. The default
            start message will be sent if not specified.
            To avoid sending a message use ``None``.
            See ``msg`` for format.
        end_msg (str or callable): Template for sending a message
            at the completion of the experiment. The default
            start message will be sent if not specified.
            To avoid sending a message use ``None``.
            See ``msg`` for format.
        error_msg (str or callable): Template for sending a message
            when an error is detected. The default
            error message will be sent if not specified.
            To avoid sending a message use ``None``.
            See ``msg`` for format.
        context_object (object): Custom object that contains data used to
            format the text or the filenames. Optional, default is ``None``.
    """

    trigger: 'TriggerLike' = (1, 'epoch')

    def __init__(
        self,
        webhook_url: str,
        msg: Union[
            str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
        ],
        *,
        start_msg: Optional[Union[
            str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
        ]] = _default_start_msg,
        end_msg: Optional[Union[
            str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
        ]] = _default_end_msg,
        error_msg: Optional[Union[
            str, Callable[[ExtensionsManagerProtocol, Any, dict, Exception], str]
        ]] = _default_error_msg,
        context_object: Optional[object] = None,
    ) -> None:

        self._webhook_url = webhook_url
        # values in current observation or log report to send to slack
        self._msg = msg
        self._start_msg = start_msg  # type: ignore[assignment]
        self._end_msg = end_msg  # type: ignore[assignment]
        self._error_msg = error_msg  # type: ignore[assignment]
        self._context = context_object

    def _send_message(
        self,
        manager: ExtensionsManagerProtocol,
        text: Union[
            str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
        ]
    ) -> None:
        observation = manager.observation
        if callable(text):
            text = text(manager, self._context, observation)
        else:
            text = text.format(
                manager=manager, context=self._context, **observation)

        if self._webhook_url:
            payload = json.dumps({'text': text}).encode('utf-8')
            request_headers = {'Content-Type': 'application/json; charset=utf-8'}
            # res = requests.post(self._webhook_url, json.dumps(payload))
            request = urllib.request.Request(
                url=self._webhook_url,
                data=payload,
                method='POST',
                headers=request_headers
            )
            urllib.request.urlopen(request)

    def initialize(self, manager: ExtensionsManagerProtocol) -> None:
        if self._start_msg is not None:
            self._send_message(manager, self._start_msg)

    def finalize(self, manager: ExtensionsManagerProtocol) -> None:
        if self._end_msg is not None:
            self._send_message(manager, self._end_msg)

    def on_error(
            self,
            manager: ExtensionsManagerProtocol,
            exc: Exception,
            tb: types.TracebackType
    ) -> None:
        if _slack_sdk_available and self._error_msg is not None:
            self._send_message(
                manager, self._error_msg, error=exc)  # type: ignore[call-arg,arg-type]

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:
        self._send_message(manager, self._msg)
