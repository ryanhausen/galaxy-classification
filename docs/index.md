---
# You don't need to edit this file, it's empty on purpose.
# Edit theme's home layout instead if you wanna make some changes
# See: https://jekyllrb.com/docs/themes/#overriding-theme-defaults
layout: default
---
{% include kanban.html %}

<div class="posts">
  {% for post in site.posts %}
    <article class="post">

      <h1><a href="{{ site.url }}{{ post.url }}">{{ post.title }}</a></h1>

      <div class="entry">
        {{ post.excerpt }}
      </div>

      <a href="{{ site.eurl }}{{ post.url }}" class="read-more">Read More</a>
    </article>
  {% endfor %}
</div>
