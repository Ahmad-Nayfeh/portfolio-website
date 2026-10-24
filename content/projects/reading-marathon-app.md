---
title: "Reading Marathon: An Android App for Reading Groups"
slug: "reading-marathon-app"
date: "2026-10-24"
coverImage: "/images/projects/reading-marathon-app.jpg"
tags: ["Android", "Kotlin", "Jetpack Compose", "Firebase", "Arabic"]
excerpt: "An Arabic Android app for reading with friends: log today's minutes in one tap, and they count in every group you belong to. Groups read a shared book, with a ranking and an activity feed. It replaced a Streamlit dashboard fed by Google Forms."
category: "Products"
status: "Released"
liveDemoUrl: "https://ahmad-nayfeh.github.io/marathon/"
featured: true
linkLabel: "Download"
gallery:
  - "/images/projects/reading-marathon-app/app-home.jpg"
  - "/images/projects/reading-marathon-app/app-group.jpg"
  - "/images/projects/reading-marathon-app/app-log.jpg"
highlights:
  - "One-tap logging, a reading timer, and quick-log buttons in the reminder"
  - "Groups with roles, invite codes and public discovery, on Firebase with tested security rules"
  - "Scoring and statistics in a pure-Kotlin module with unit tests"
  - "Imports the group's old Google Form history"
---

# Reading Marathon (ماراثون القراءة)

> The harder it is to log reading, the fewer people do it. So logging takes one tap, and everything else is built around that.

**Download:** [ahmad-nayfeh.github.io/marathon](https://ahmad-nayfeh.github.io/marathon/) (Android 8+, free)

## What it does

- **Logging in one tap:** buttons add 15, 30, 45 or 60 minutes to today, with undo. A timer runs in a notification while you read, and the evening reminder has quick-log buttons, so a day can be logged without opening the app. Any of the last 14 days can be edited.
- **Reading is personal, groups add it up:** a reader logs once, and the minutes count in every group they are in.
- **Groups:** for friends, family or a reading club. Private (invite code or link) or public (anyone can find and join). Roles: owner, admins, members.
- **A shared book:** each period has a book and dates, with the ranking, who read today, and who finished the book.
- **Points:** one point per N minutes (10 by default) plus a bonus for finishing the book (50 by default), set per group. The ranking can be shared as text to WhatsApp.
- **Personal page:** a daily goal, streaks, hours this month and year, and a reading heatmap.
- **For admins:** books, members, readers without the app (an admin logs for them, and merges their history into their account if they join later), import of the old Google Form responses, and CSV export.

<div class="not-prose grid grid-cols-2 gap-4 sm:grid-cols-3">
  <img src="/images/projects/reading-marathon-app/app-log.jpg" alt="Editing a day of reading" loading="lazy" />
  <img src="/images/projects/reading-marathon-app/app-activity.jpg" alt="A group's activity feed" loading="lazy" />
  <img src="/images/projects/reading-marathon-app/app-groups.jpg" alt="My groups and public groups" loading="lazy" />
</div>

## How it is built

- **Kotlin + Jetpack Compose** with Material 3, DataStore and WorkManager. The interface is Arabic and always right-to-left, with a calm paper-and-ink design.
- **A pure-Kotlin `core` module** holds every rule: points, rankings, streaks, validation and the CSV import, with unit tests.
- **Firebase** (Google or anonymous sign-in, Firestore with offline cache). A day is written to the reader's private log and to each of their groups. The security rules enforce roles, joining by invite code and who may write what; 70 tests run them against the Firestore emulator.
- **Testing:** a screenshot tour of every main screen, rendered with Robolectric on an in-memory backend, runs in CI without an emulator.
- **CI/CD:** GitHub Actions builds the app and publishes signed releases from version tags.

## Where it came from

The first version (2025) was a **Streamlit dashboard**: readers filled in a Google Form, and the organiser's dashboard pulled the responses through the Google Sheets API into Firestore and drew the charts with Plotly. It worked, but logging meant filling a form every day, and readers never saw the results unless someone shared a screenshot.

The Android app's first release kept that model (one organiser, a long daily form). Version 2 rebuilt it around one-tap logging and groups, after watching how often people actually logged. The web version still runs at [reading-marathon.streamlit.app](https://reading-marathon.streamlit.app/) ([privacy policy](/privacy), [terms](/terms)).
