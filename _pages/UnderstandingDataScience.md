---
layout: single
permalink: /comprehension/
title: "Data Science Theory Comprehension"
author_profile: true
header:
  image: "/pictures/TokyoekiRes.jpg"
---

{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}
