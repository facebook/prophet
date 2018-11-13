{%- extends 'markdown.tpl' -%}

{%- block header -%}
---
layout: docs
docid: "{{resources['metadata']['name']}}"
title: "{{resources['metadata']['name'].replace('_', ' ').title()}}"
permalink: /docs/{{resources['metadata']['name']}}.html
subsections:
{%- for cell in nb['cells'] if cell.cell_type == 'markdown' and '##' in cell.source -%}
{% for line in cell.source.split('\n') if line.startswith('##') %}
  - title: {{ line.lstrip('# ') }}
    id: {{ line.lstrip('# ').lower().replace(' ', '-') }}
{%- endfor -%}
{% endfor %}
---
{%- endblock header -%}

{%- block any_cell -%}
{%- if not cell.metadata.get("block_hidden", False) -%}
    {{ super() }}
{%- endif -%}
{%- endblock any_cell -%}

{% block input %}
{%- if cell.source[:3] == "%%R" -%}
```R
# R
{{ '\n'.join(cell.source.split('\n')[1:]) }}
```
{%- else -%}
```python
# Python
{{ cell.source }}
```
{%- endif -%}
{%- endblock input -%}

{%- block output_group -%}
{%- if not cell.metadata.get("output_hidden", False) -%}
    {{ super() }}
{%- endif -%}
{%- endblock output_group -%}

{%- block input_group -%}
{%- if not cell.metadata.get("input_hidden", False) -%}
    {{ super() }}
{%- endif -%}
{%- endblock input_group -%}

{% block data_png %} 
![png](/prophet/static/{{ output.metadata.filenames['image/png'] }}) 
{% endblock data_png %}

{% block markdowncell %}
{%- set lines = cell.source.split('\n') -%}
{%- for line in lines -%}
{% if line.startswith('##') %}
<a id="{{ line.lstrip('# ').lower().replace(' ', '-') }}"> </a>
{% endif %}
{{ line }}
{% endfor %}
{% endblock markdowncell %}