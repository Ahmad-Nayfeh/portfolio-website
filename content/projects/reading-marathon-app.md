---
title: "Reading Marathon: An Android App for Reading Groups"
slug: "reading-marathon-app"
date: "2026-10-24"
coverImage: "/images/projects/reading-marathon-app.jpg"
tags: ["Android", "Kotlin", "Jetpack Compose", "Firebase", "Arabic"]
excerpt: "An Arabic Android app for reading, alone or with friends: pick the book you are reading, log your time in it with one tap, and keep a personal shelf. Groups read a shared book together, with a ranking. It replaced a Streamlit dashboard fed by Google Forms."
category: "Products"
status: "Released"
liveDemoUrl: "https://ahmad-nayfeh.github.io/marathon/"
featured: true
linkLabel: "Download"
gallery:
  - "/images/projects/reading-marathon-app/app-home.jpg"
  - "/images/projects/reading-marathon-app/app-shelf.jpg"
  - "/images/projects/reading-marathon-app/app-group.jpg"
highlights:
  - "Every minute is logged against a book: one tap, a timer, or the reminder"
  - "A catalog of about 4,900 books from Wikidata, searchable offline"
  - "Groups with roles and invite codes, on Firebase with tested security rules"
  - "Export your data or delete your account from inside the app"
role: "Designed, built, and released it on my own"
result: "Released for Android; replaced a Google Form and a Streamlit dashboard"
---

# Reading Marathon (ماراثون القراءة)

> The harder it is to log reading, the fewer people do it. So logging takes one tap, and everything else is built around that.

**Download:** [ahmad-nayfeh.github.io/marathon](https://ahmad-nayfeh.github.io/marathon/) (Android 8+, free)

## What it does

- **The book comes first:** every minute is logged against a book. The home screen has a card for each book you are reading, with buttons for 15, 30, 45 or 60 minutes (with undo), a reading timer that runs in a notification, and "other time" for any of the last 14 days. The evening reminder logs into your book without opening the app.
- **My shelf:** your books, with or without a group: reading now, finished, and want to read, with your time in each.
- **A real catalog:** about 4,900 books, each with its Arabic title, its author, and its English title as another name. The list comes from Wikidata (written works with an Arabic Wikipedia article and a known author) and is searched on the phone, so it works offline. Missing books come from Google Books or are typed in, and readers can suggest fixes.
- **Groups:** private (invite code or link) or public. Each group reads a shared book over set dates, with a ranking, who read today, and who finished. The group chooses whether the ranking counts all reading or only the shared book.
- **Everyone logs for themselves:** nobody can log reading or a finish for anyone else, not even a group owner. The security rules enforce it.
- **Your data is yours:** export your reading log and shelf as CSV, or delete your account and everything in it, from inside the app.

<div class="not-prose grid grid-cols-2 gap-4 sm:grid-cols-3">
  <img src="/images/projects/reading-marathon-app/app-log.jpg" alt="Logging time for a book on any of the last two weeks" loading="lazy" />
  <img src="/images/projects/reading-marathon-app/app-catalog.jpg" alt="Finding a book in the catalog" loading="lazy" />
  <img src="/images/projects/reading-marathon-app/app-activity.jpg" alt="A group's activity" loading="lazy" />
</div>

## How it is built

- **Kotlin + Jetpack Compose** with Material 3, DataStore and WorkManager. The interface is Arabic and always right-to-left, with a calm paper-and-ink design in light and dark.
- **A pure-Kotlin `core` module** holds every rule: points, rankings, streaks, validation and catalog search, with unit tests.
- **Firebase:** Google sign-in only, and Firestore with an offline cache. A reading is written to the reader's private log and shelf, and to each of their groups. The security rules enforce roles, invite codes, and that each reader writes only their own reading; 127 tests run them against the Firestore emulator.
- **Admin panel** in the app and on the web: catalog review, groups, users, announcements, and an audit log of every action.
- **Testing:** a screenshot tour of the main screens, rendered with Robolectric on an in-memory backend, runs without an emulator.

## Where it came from

The first version (2025) was a **Streamlit dashboard**: readers filled in a Google Form, and the organiser's dashboard pulled the responses through the Google Sheets API into Firestore and drew the charts with Plotly. It worked, but logging meant filling a form every day, and readers never saw the results unless someone shared a screenshot. The Android app replaced it with one-tap logging, books, and groups.
