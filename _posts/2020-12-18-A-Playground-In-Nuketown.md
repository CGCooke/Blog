---
toc: true
layout: post
description: An overview of a new family of posts.
categories: [Computer Vision, Nuketown84]
image: images/2020-12-18-A-Playground-In-Nuketown/header.jpg
---

Computer vision is a powerful way to learn more about the world around us. Deep Learning is powering a renaissance in the field, and is unlocking powerful new capabilities in the field of *SLAM* (Simultaneous Localisation and Mapping). I want to spend some time exploring this area, and as part of this, I want to work with an interesting source of data, that's visually interesting. 

One interesting source of data is video games, and I've settled on the fast, frantic, mega franchise *Call of Duty: Black Ops Cold War*. [Nuketown84](https://callofduty.fandom.com/wiki/Nuketown_%2784) is one of the maps/levels, which presents a gritty, decaying ambiance, perfect for learning more about how we can apply computer vision in practice. You watch see some of the action [here](https://www.youtube.com/watch?v=dozMeWeraFk).

If our end goal is to recognise where we are in the world, based on what we can see, then our starting point, is having some ground truth data (where we actually are). Capturing this ground truth data, is in itself, an interesting challenge, which I'm going to slowly build up over the next couple of posts.

One strategy, is to infer the players position from both the mini map, as well as the on screen compass. By combining different measurements over time, we should be able to understand where the player is in the world. 



![_config.yml]({{ site.baseurl }}/images/2020-12-18-A-Playground-In-Nuketown/Nuketown-84-1.jpg)




