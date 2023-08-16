# WhatsApp Web Packet Analysis Repository

This is our WhatsApp Web Packet Analysis project, inside this repository you will discover an assortment of code and resources meticulously crafted for the investigation of packets originating from group messages on WhatsApp's web platform. 
This project entails packet sniffing, discerning message-associated packets, and crafting illuminating visualizations to expedite packet differentiation.  
The analytical journey unfolds in two distinctive phases: 
• The attacked user is always active in (at most) a single IM group.
• The attacked user may be active in several IM groups simultaneously. 
יעיייעעעכ
## Table of Contents

- [Introduction](#introduction)
- [Project Layout](#project-layout)
- [Packet Examination](#packet-examination)
  - [Segment 1: Incorporating Filters](#segment-1-incorporating-filters)
  - [Segment 2: Devoid of Filters and Accompanied by Ambient Music](#segment-2-devoid-of-filters-and-accompanied-by-ambient-music)
- [Visual Depictions](#visual-depictions)
- [Resource Repository](#resource-repository)
- [Code Repository](#code-repository)
- [Participation](#participation)
- [License](#license)

## Introduction

The fundamental aim of the WhatsApp Web Packet Analysis project is to meticulously dissect and decode packets generated by group messages on WhatsApp's web platform. By deploying an array of analytical methodologies and innovative filtration strategies, the project endeavors to identify packets intimately associated with dispatched messages. Visual aids are harnessed to expedite the intricate process of packet differentiation.

## Project Layout

The project architecture is carefully organized into three key domains:

- `res`: Within this section, you'll find an array of graphical representations, each an outcome of the analytical process. These visuals encompass depictions that unravel the complex interplay between time and packet size, graphical insights into the probability density function (PDF) of packet sizes, and visualizations that articulate the complementary cumulative distribution function (CCDF) of packet sizes.
- `resources`: This repository houses eight pcapng files and eight CSV files, meticulously categorized into four for filtered packets and another four for unfiltered packets. These resources serve as the bedrock for the exhaustive analysis.
- `src`: Nestled within this domain are four Python scripts, each designed to orchestrate a unique category of visual representation: CCDF plots, PDF plots, and elucidating visualizations that unravel the interplay between time and packet size.

## Packet Examination

### Segment 1: Incorporating Filters

During this pivotal phase, packet examination unfolds with meticulous precision through the incorporation of a dedicated filter targeting `tcp.port 443`. This filter serves as a crucial instrument in eliminating extraneous noise, thus accentuating the focus on pertinent packets. The primary objective here is the isolation of packets intricately linked with dispatched messages, accompanied by the creation of visual depictions designed to expedite their recognition.

### Segment 2: Devoid of Filters and Accompanied by Ambient Music

The second phase of packet examination eschews packet filters altogether, introducing an authentic layer of ambiance through the melodious backdrop of real-world noise. This auditory tapestry echoes the experience of YouTube melodies playing subtly in the background. The core objective in this phase is to assess the influence of real-world noise on the accuracy of packet analysis.

## Visual Depictions

Housed within the `res` section are a diverse array of graphical representations, each providing profound insights into the intricate process of packet analysis:

- **Temporal Patterns vs. Packet Size**: These graphical representations cast a spotlight on the complex interplay between time and packet size, revealing underlying patterns and tendencies.
- **Graphs of Probability Density Function (PDF)**: PDF graphs artistically capture the essence of the probability distribution of packet sizes. A total of eight graphical masterpieces are meticulously generated, addressing both filtered and unfiltered packets.
- **Complementary Cumulative Distribution Function (CCDF) Graphs**: The CCDF graphs serve as visual orchestrations depicting the complementary cumulative distribution function of packet sizes. These graphs exclusively address the four packets that have undergone the meticulous filtration process.

## Resource Repository

The `resources` section serves as the treasury of foundational materials essential for orchestrating the in-depth analysis. This treasury encompasses eight CSV files, encapsulating the nuanced essence of both meticulously filtered and unfiltered packets.

## Code Repository

Within the `src` section, you'll discover the quintessential Python code required for orchestrating a diverse ensemble of visual representations. The four Python scripts are meticulously crafted to harmoniously generate CCDF plots, PDF plots, and intricately orchestrated visualizations that poetically articulate the delicate interplay between time and packet size.

## Participation

Your contributions to this endeavor are both valued and cherished! To contribute, we humbly request your adherence to the following guidelines:

1. Fork the repository.
2. Create a new branch for your envisioned feature or bug fix: `git checkout -b feature-name`.
3. Enact your modifications and elegantly commit them: `git commit -m 'Articulate a visionary feature'`.
4. Elegantly push your alterations to the designated branch: `git push origin feature-name`.
5. Initiate a pull request meticulously narrating your transformative changes.

Please ensure your contributions align seamlessly with the meticulously set coding standards and guidelines.

We extend our heartfelt gratitude for your engagement and commitment to the WhatsApp Web Packet Analysis project!
