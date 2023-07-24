{{ fullname | escape | underline}}

.. automodule:: {{ fullname }}

   {% block attributes %}
   {% if attributes %}
   .. rubric:: {{ _('Module Attributes') }}

   .. autosummary::
      :toctree: .

      {% for item in attributes %}
         {{ fullname }}.{{ item }}
      {%- endfor %}
      {% endif %}
      {% endblock %}

      {% block functions %}
      {% if functions %}
   .. rubric:: {{ _('Functions') }}

   .. autosummary::
      :toctree: .

      {% for item in functions %}
         {{ fullname }}.{{ item }}
      {%- endfor %}
      {% endif %}
      {% endblock %}

      {% block classes %}
      {% if classes %}
   .. rubric:: {{ _('Classes') }}

   .. autosummary::
      :toctree: .

      {% for item in classes %}
         {{ fullname }}.{{ item }}
      {%- endfor %}
      {% endif %}
      {% endblock %}

      {% block exceptions %}
      {% if exceptions %}
   .. rubric:: {{ _('Exceptions') }}

   .. autosummary::
      :toctree: .

      {% for item in exceptions %}
         {{ fullname }}.{{ item }}
      {%- endfor %}
      {% endif %}
      {% endblock %}

{% block modules %}
{% if modules %}
.. rubric:: Modules

.. autosummary::
   :toctree: .
   :recursive:
   {% for item in modules %}
      {{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}