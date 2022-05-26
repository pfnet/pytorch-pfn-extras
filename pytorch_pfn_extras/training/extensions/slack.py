import os
import json
import warnings

from typing import Any, Callable, List, Optional, Union

from pytorch_pfn_extras.training import extension
from pytorch_pfn_extras.training._manager_protocol import ExtensionsManagerProtocol

try:
    import requests  # type: ignore[import]
    _requests_available = True
except ImportError:
    _requests_available = False

try:
    import slack_sdk
    _slack_sdk_available = True
except ImportError:
    _slack_sdk_available = False


class Slack(extension.Extension):
    """An extension to communicate with Slack.

    This extension receives a ``text_template`` argument that contains
    placeholders that will be populated with ``manager`` or ``manager.observation``
    fields, and custom objects such as ``context_object``.

    For example. `text_template = "Loss {loss} for iteration {.iteration}"`
    will be populated as `text_template.format(manager, context_object, **observation)`
    retrieving the ``loss`` value from the ``observation`` and ``.iteration`` from
    the manager object. Instead of string, a callable object taking the
    ``ExtensionsManager``, ``context_object`` and the observation dictionary can
    be used instead. The same rules apply for the ``filenames_template`` argument.

    Args:
        channel_id (str): The channel where messages or files will be sent.
        text_template (str or callable): Template for sending the message.
            It can be a string to be formatted using ``.format`` or a callable
            that returns a string. In both cases, `manager`, the current
            observation dict and the context object are used to look up
            values of interest.
        filenames_template (list of str or callable): list of files that will
            be uploaded to slack, these are string templates that can take
            values in the same way as ``text_template``. Optional.
        use_threads (bool): If subsequent calls of this extensions should be
            posted as a thread of the original message or not.
            Default is ``False``.
        context_object (object): Custom object that contains data used to
            format the text or the filenames. Optional, default is ``None``.
        webhook_url (str): Webhook to send messages, it is mutually exclusive
            with ``token`` and ``client``. Used when slack_sdk is not available.
            Optional, default is ``None``.
        token (str): Token for the slack api, if ``None`` the environment
            variable ``SLACK_TOKEN`` will be used. Ignored if ``client`` is
            supplied. Optional, default is ``None``.
        client (slack_sdk.WebClient): In case that there is an already created
            slack client in the application, allows to directly use it.
            Optional, default is ``None``
    """
    def __init__(
        self,
        channel_id: str,
        text_template: Union[
            str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
        ],
        filenames_template: Optional[
            List[
                Union[
                    str, Callable[[ExtensionsManagerProtocol, Any, dict], str]
                ]
            ]
        ] = None,
        use_threads: bool = False,
        context_object: Optional[object] = None,
        webhook_url: Optional[str] = None,
        token: Optional[str] = None,
        client: Optional[Any] = None,  # slack_sdk.WebClient, Any to avoid mypy errors
    ) -> None:
        if not _slack_sdk_available and webhook_url is None:
            warnings.warn(
                '`slack_api` package is unavailable. '
                'Slack will do nothing unless a webhook url is provided.')
            return

        if not _requests_available and webhook_url is not None:
            warnings.warn(
                '`requests` package is unavailable. '
                'Slack will do nothing.')
            return

        if webhook_url is None:
            self._client = client
            if client is None:
                if token is None:
                    token = os.environ.get('SLACK_TOKEN', None)
                if token is None:
                    raise RuntimeError(
                        '`token` is needed for communicating with slack')
                self._client = slack_sdk.WebClient(token=token)
        else:
            if (token, client) != (None, None):
                raise ValueError(
                    '`webhook_url` can\'t be used with client and token')
            if filenames_template is not None:
                raise ValueError(
                    '`webhook_url` can\'t be used to post files'
                    'please install `slack_sdk`')
            self._client = None
        self._webhook_url = webhook_url
        # values in current observation or log report to send to slack
        self._text = text_template
        if filenames_template is None:
            filenames_template = []
        self._filenames = filenames_template
        self._context = context_object
        self._channel_id = channel_id
        self._use_threads = use_threads
        self._ts = None

    def __call__(self, manager: ExtensionsManagerProtocol) -> None:

        observation = manager.observation
        if callable(self._text):
            text = self._text(manager, self._context, observation)
        else:
            text = self._text.format(manager, self._context, **observation)

        if self._webhook_url and _requests_available:
            payload = {'text': text}
            res = requests.post(self._webhook_url, json.dumps(payload))
            if res.status_code != 200:
                raise Exception(res.status_code, res.text)
            return
        elif not _requests_available:
            return

        if not _slack_sdk_available:
            return

        assert self._client is not None

        ts = None
        if self._use_threads:
            ts = self._ts
        response = self._client.chat_postMessage(
            channel=self._channel_id,
            text=text,
            thread_ts=ts,
        )
        assert response.get("ok")  # type: ignore[no-untyped-call]
        if self._use_threads and ts is None:
            ts = response.get("ts")  # type: ignore[no-untyped-call]
            self._ts = ts
        for filename in self._filenames:
            if callable(filename):
                filename = filename(manager, self._context, observation)
            else:
                filename = filename.format(
                    manager, self._context, **observation)
            response = self._client.files_upload(
                channels=self._channel_id,
                file=filename,
                thread_ts=ts,
            )
            assert response.get("ok")  # type: ignore[no-untyped-call]
