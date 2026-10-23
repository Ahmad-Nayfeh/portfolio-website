---
title: "Khatwa: A Walking App That Plays Like a Game"
slug: "khatwa"
date: "2026-10-11"
coverImage: "/images/projects/khatwa.jpg"
tags: ["Product", "Android", "Kotlin", "Arabic"]
excerpt: "A native Android app that counts steps, can lock the phone until you walk, and turns walking into daily quests, levels, and streaks, with walk suggestions timed around weather and prayer times."
category: "Products"
featured: true
---

# Khatwa (خطوة)

> I wanted to walk more, and willpower was not enough. Khatwa makes skipping a walk cost something.

## What it does

- **Step and distance tracking** from the phone's hardware step sensor, working offline.
- **Phone lock:** optionally locks the phone until a daily step target is reached, with an allowlist for essential apps.
- **The "System":** daily quests, experience points, ranks from E to S, and a seven-tier streak flame.
- **Daily walk challenge:** suggests a nearby place to walk, at the best time for the weather and prayer times (Saudi cities), using OpenStreetMap and Open-Meteo.
- **Walk maps** of every session and the streets you have explored, plus a six-month bodyweight training program (Arabic and English).
- **Optional account** with private cloud backup and groups with leaderboards.

<div style="display:flex;gap:12px;flex-wrap:wrap;justify-content:center">
  <img src="/images/projects/khatwa/home.jpg" alt="Home" width="220" />
  <img src="/images/projects/khatwa/system.jpg" alt="The System" width="220" />
  <img src="/images/projects/khatwa/my-map.jpg" alt="Walk map" width="220" />
</div>

## How it is built

- **Kotlin + Jetpack Compose**, Room, and WorkManager, with the core logic in a pure-Kotlin module tested without the Android SDK.
- **Firebase** (Auth, Firestore with tested security rules, App Check) for the optional account and groups.
- **CI/CD:** GitHub Actions runs unit and emulator tests and publishes signed releases.
- Fully bilingual, right-to-left first.
