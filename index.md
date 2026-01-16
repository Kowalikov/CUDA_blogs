---
layout: default
title: Home
---

<ul class="sidebar-nav">
  {% for page in site.pages %}
    {% if page.title %}
      <li><a class="sidebar-nav-item{% if page.url == page.url %} active{% endif %}" href="{{ page.url | relative_url }}">{{ page.title }}</a></li>
    {% endif %}
  {% endfor %}
</ul>

<!-- existing homepage content -->