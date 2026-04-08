---
layout: default
title: "The cuDNN Blog"
---

# Welcome to The cuDNN Blog

Hey there, fellow GPU enthusiast! This is the unofficial-feeling, totally-official corner of the internet where we talk about **cuDNN** — NVIDIA's library for accelerating deep learning primitives on GPUs.

Whether you're training a massive transformer, fine-tuning a convolutional network, or just trying to get GPUs to go *brrr*, cuDNN is the engine under the hood making it happen. This blog is where we share release notes, installation guides, and the occasional deep-dive into what makes cuDNN tick.

Check out the **Installation Guides** in the sidebar to get started, or read through the latest release notes below.

---

## Latest Releases

<ul class="post-list">
{% for post in site.posts %}
  <li>
    <div class="post-card">
      <span class="card-meta">{{ post.date | date: "%B %d, %Y" }}</span>
      <h2><a href="{{ post.url | relative_url }}">{{ post.title }}</a></h2>
      <p>{{ post.description }}</p>
    </div>
  </li>
{% endfor %}
</ul>
