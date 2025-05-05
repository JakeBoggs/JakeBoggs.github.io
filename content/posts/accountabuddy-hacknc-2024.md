---
title: "AccountaBuddy: Your AI Accountability Partner - HackNC 2024"
date: "2024-11-04"
draft: false
---
Check out the project on **[Devpost](https://devpost.com/software/nudge-4groy0)** and view the source code on **[GitHub](https://github.com/JakeBoggs/AccountaBuddy-HackNC2024)**.

{{< youtube YLjw9UnhMEI >}}

I spent a weekend at HackNC building something to help manage all of my other side projects. What started as a simple "Wouldn't it be cool if an AI actually rang you to ask if you'd done your work?" turned into AccountaBuddy, a lightweight app that does more than fire off push notifications - it actually calls you, celebrates your wins, and helps you problem-solve when tasks stall.

## The Why: When Reminders Aren't Enough

I'm not the only one who's ever ignored yet another notification. A to-do list can feel like a list of obligations, or worse, just noise. AccountaBuddy was born from the idea that human accountability is powerful - so why not let technology step in as a friendly collaborator? Instead of sending one-way reminders, it reaches out, listens to you, and adapts to your situation.

## What It Does, in Practice

*   **Task creation & scheduling:** Sign up, add your tasks, and pick check-in times.
*   **Real-time voice calls:** At each scheduled moment, AccountaBuddy actually rings your phone via [bland.ai](https://www.bland.ai/).
*   **Dynamic follow-up:** If you've completed a task, it cheers you on. If not, it asks what's blocking you and suggests next steps.
*   **Interactive planning:** You can reprioritize or break a big goal into bite-sized actions in an on-the-fly chat.

Rather than a static list of "done" or "not done," it turns to-dos into a conversation. The UI was inspired by our code editor, Cursor, with the AI proposing modifications to the to-do list and the user confirming or denying them.

## Under the Hood

Despite having just two developers and 24 hours, we got a working proof of concept by combining:

*   **Next.js + React** for the UI and routing
*   **Prisma** for defining and managing users and tasks in our database
*   **bland.ai** for dialing out and handling call flows
*   **ChatGPT** to guide the conversation, interpret responses, and offer personalized suggestions

## What We Struggled With

*   **State management:** React's component lifecycle can be tricky when you need live updates for tasks that might change status mid-call.
*   **API choreography:** We had to make sure calls, chat logic, and database updates stayed in sync without visible delays.
*   **Background jobs:** Scheduling outbound calls reliably—especially under a tight timeline—meant building a small worker queue and retry logic.

## Takeaways & Next Steps

By the end of the hackathon, AccountaBuddy was already helping us plan the remaining tasks, but there's still more we'd like to do in the future:

*   **Stripe integration** for a lightweight subscription model
*   **Live voice conversations** powered by ChatGPT Advanced Voice Mode, so you can brainstorm out loud with your AI buddy in real time
*   **Deeper analytics,** to show you trends in your productivity rather than just individual tasks