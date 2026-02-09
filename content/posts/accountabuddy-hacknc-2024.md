---
title: "AccountaBuddy: An AI Accountability Partner - HackNC 2024"
date: "2024-11-04"
draft: false
summary: "Weekend hackathon project that calls your phone to check on your tasks."
---
Check out the project on **[Devpost](https://devpost.com/software/nudge-4groy0)** and view the source code on **[GitHub](https://github.com/JakeBoggs/AccountaBuddy-HackNC2024)**.

I spent a weekend at HackNC building something to help manage all of my other side projects. What started as "Wouldn't it be cool if an AI actually rang you to ask if you'd done your work?" turned into AccountaBuddy, a app that does more than fire off push notifications - it actually calls you, celebrates your wins, and helps you problem-solve when progress stalls.

{{< youtube YLjw9UnhMEI >}}

We aren't not the only ones who've ever ignored yet another notification. A to-do list can feel like a list of obligations, or worse, just noise. AccountaBuddy was born from the idea that human accountability is powerful - so why not let technology step in as a friendly collaborator? Instead of sending one-way reminders, it reaches out, listens to you, and adapts to your situation.

## Features

*   **Task creation & scheduling:** Sign up, add your tasks, and pick check-in times.
*   **Real-time voice calls:** At each scheduled check-in, AccountaBuddy actually rings your phone via [bland.ai](https://www.bland.ai/).
*   **Dynamic follow-up:** If you've completed a task, it cheers you on. If not, it asks what's blocking you and suggests next steps.
*   **Interactive planning:** You can reprioritize or break a big goal into bite-sized actions in an on-the-fly chat.

Rather than a static list of "done" or "not done," it turns to-dos into a conversation. The UI was inspired by our code editor, Cursor, with the AI proposing modifications to the to-do list and the user accepting or rejecting them.

Despite having just two developers and 24 hours, we got a working proof of concept by combining:

*   **Next.js + React** for the UI and routing
*   **Prisma** for managing objects in our database
*   **bland.ai** for dialing out and handling call flows
*   **GPT-4o** to power the conversations through the OpenAI API

By the end of the hackathon, AccountaBuddy was already helping us plan the remaining tasks, but there's still more we'd like to do in the future:

*   **Stripe integration** and a subscription model
*   **Live voice conversations** for brainstorming and planning, not just check-ins
*   **Analytics** to show trends in your productivity