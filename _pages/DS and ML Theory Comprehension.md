---
layout: single
permalink: /machine-learning/
title: "Data Science and ML Comprehension:"
author_profile: true
header:
  image: "/pictures/TokyoekiRes.jpg"
mathjax: "true"
toc: true
toc_label: "Projects"
toc_icon: "cog"
toc_sticky: true
---

{% include group-by-array collection=site.posts field="tags" %}

{% for tag in group_names %}
  {% assign posts = group_items[forloop.index0] %}
  <h2 id="{{ tag | slugify }}" class="archive__subtitle">{{ tag }}</h2>
  {% for post in posts %}
    {% include archive-single.html %}
  {% endfor %}
{% endfor %}
