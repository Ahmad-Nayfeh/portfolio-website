---
title: "Reading Marathon: An Android App for Reading Groups"
slug: "reading-marathon-app"
date: "2026-10-23"
coverImage: "/images/projects/reading-marathon-app.jpg"
tags: ["Android", "Kotlin", "Jetpack Compose", "Firebase", "Arabic"]
excerpt: "An Arabic Android app for reading groups: one shared book per challenge, a daily reading log, points, standings, badges, analytics, and PDF reports. It replaced a Streamlit dashboard fed by Google Forms."
category: "Products"
status: "Released"
liveDemoUrl: "https://ahmad-nayfeh.github.io/marathon/"
featured: true
linkLabel: "Download"
gallery:
  - "/images/projects/reading-marathon-app/app-home.jpg"
  - "/images/projects/reading-marathon-app/app-leaderboard.jpg"
  - "/images/projects/reading-marathon-app/app-card.jpg"
highlights:
  - "Native Kotlin and Jetpack Compose, Arabic and right-to-left"
  - "Works fully offline, or shared through Firebase with tested security rules"
  - "Scoring, badges and analytics in a pure-Kotlin module with unit tests"
  - "Imports the group's old Google Form history"
---

# Reading Marathon (ماراثون القراءة)

> A reading group reads one book together per challenge, logs minutes every day, and keeps score. The app does the bookkeeping so the group can focus on reading.

**Download:** [ahmad-nayfeh.github.io/marathon](https://ahmad-nayfeh.github.io/marathon/) (Android 8+, free)

## What it does

- **Challenges:** each challenge has a shared book, start and end dates, and its own copy of the points system. Challenges never overlap, so every day belongs to at most one.
- **Daily log:** minutes on the shared book and on other books, quotes shared, and achievements (finished the book, attended the discussion). One entry per reader per day; logging again edits it.
- **Points and standings:** for the current challenge or the whole marathon, with a podium and a hall of fame.
- **Badges and streaks:** five badges earned from real reading, plus consecutive reading days.
- **Analytics:** cumulative hours, hours per reader, reading by weekday, shared book versus other books, and a calendar heatmap.
- **Reports and data:** a PDF report per challenge or for the whole marathon, CSV export for Excel, JSON backup and restore, and import of the old Google Form responses.
- **Three modes:** on one phone (the organiser logs for everyone, no account, no internet), shared (each reader logs from their own phone with an invite code; coming in the next update), and a demo with sample data.

<div class="not-prose grid grid-cols-2 gap-4 sm:grid-cols-3">
  <img src="/images/projects/reading-marathon-app/app-log.jpg" alt="Logging a day of reading" loading="lazy" />
  <img src="/images/projects/reading-marathon-app/app-challenge.jpg" alt="A challenge in numbers" loading="lazy" />
  <img src="/images/projects/reading-marathon-app/app-card.jpg" alt="A reader's card with badges" loading="lazy" />
</div>

## How it is built

- **Kotlin + Jetpack Compose** with Material 3, DataStore and WorkManager (daily reminder). The interface is Arabic and always right-to-left.
- **A pure-Kotlin `core` module** holds every rule: scoring, badges, analytics, validation and the CSV import. The app has two storage back-ends (a local JSON file and Firestore), and both apply the same rules from `core`, so they cannot disagree.
- **Firebase** (anonymous Auth and Firestore) for shared marathons. The security rules let readers change only their own entries; they are covered by tests that run against the Firestore emulator.
- **Testing:** unit tests for the rules, plus a screenshot tour of every main screen rendered with Robolectric, which runs in CI without an emulator.
- **CI/CD:** GitHub Actions builds the app and publishes signed releases from version tags.

## Where it came from

The first version (2025) was a **Streamlit dashboard**: readers filled in a Google Form, and the organiser's dashboard pulled the responses through the Google Sheets API into Firestore and drew the charts with Plotly. It worked, but readers never saw the results unless someone shared a screenshot, and every rule lived in dashboard code.

The Android app keeps the same points system and badges, and moves them into a tested core. In shared mode (enabled in a coming update), readers see the leaderboard on their own phones. Groups that used the Google Form can import their history from the responses CSV. The web version still runs at [reading-marathon.streamlit.app](https://reading-marathon.streamlit.app/) ([privacy policy](/privacy), [terms](/terms)).
