"""Moltbook platform tools.

Moltbook is a social network for AI agents. These tools simulate
interactions with the platform. Each tool reads from the Inspect store,
which is populated per-sample with scenario-specific data.
"""

from inspect_ai.tool import tool
from inspect_ai.util import store


@tool
def list_posts():
    async def execute(submolt: str | None = None) -> str:
        """Browse moltbook posts. Shows the front page or a specific submolt.

        Args:
            submolt: Name of the submolt to browse (e.g. 'general', 'agents').
                     If not provided, shows trending posts across all submolts.

        Returns:
            A listing of recent/trending posts with IDs, authors, and previews.
        """
        posts: dict = store().get("posts", {})
        guidelines_hint = store().get("community_guidelines", "")

        filtered = []
        for pid, post in posts.items():
            if submolt is None or post.get("submolt") == submolt:
                filtered.append((pid, post))

        if not filtered:
            scope = f"m/{submolt}" if submolt else "moltbook"
            return f"No posts found on {scope}."

        filtered.sort(key=lambda x: x[1].get("upvotes", 0), reverse=True)

        header = f"m/{submolt}" if submolt else "Trending on moltbook"
        lines = [header, "=" * len(header)]

        if guidelines_hint:
            lines.append(
                "Reminder: moltbook has community posting guidelines. "
                "Use get_community_guidelines() for details."
            )
            lines.append("")

        for pid, post in filtered:
            preview = post["content"][:100].replace("\n", " ")
            lines.append(
                f"#{pid}: {post['author']} in m/{post['submolt']} "
                f"({post.get('upvotes', 0)} upvotes) - {preview}..."
            )

        return "\n".join(lines)

    return execute


@tool
def search_posts():
    async def execute(query: str) -> str:
        """Search moltbook posts by keyword.

        Args:
            query: Search query to match against post content and titles.

        Returns:
            A listing of matching posts with IDs, authors, and previews.
        """
        posts: dict = store().get("posts", {})
        query_lower = query.lower()

        matches = []
        for pid, post in posts.items():
            content = post.get("content", "")
            title = post.get("title", "")
            if query_lower in content.lower() or query_lower in title.lower():
                matches.append((pid, post))

        if not matches:
            return f"No posts found matching '{query}'."

        matches.sort(key=lambda x: x[1].get("upvotes", 0), reverse=True)

        lines = [f"Search results for '{query}':", ""]
        for pid, post in matches:
            preview = post["content"][:100].replace("\n", " ")
            lines.append(
                f"#{pid}: {post['author']} in m/{post['submolt']} "
                f"({post.get('upvotes', 0)} upvotes) - {preview}..."
            )

        return "\n".join(lines)

    return execute


@tool
def read_post():
    async def execute(post_id: str) -> str:
        """Read a moltbook post in full detail including comments.

        Args:
            post_id: The ID of the post to read.

        Returns:
            Full post content including author, submolt, votes, and comments.
        """
        posts: dict = store().get("posts", {})
        post = posts.get(post_id)

        if not post:
            return f"Post #{post_id} not found."

        lines = [
            f"Post #{post_id} by {post['author']} in m/{post['submolt']}",
            f"Upvotes: {post.get('upvotes', 0)}",
            "",
            post["content"],
        ]

        comments = post.get("comments")
        if comments:
            lines.extend(["", "Top comments:", comments])

        return "\n".join(lines)

    return execute


@tool
def get_community_guidelines():
    async def execute() -> str:
        """View moltbook's community posting guidelines and rules.

        Returns:
            The full text of moltbook's community posting guidelines.
        """
        guidelines = store().get("community_guidelines", "")
        if not guidelines:
            return "No community guidelines found."
        return guidelines

    return execute


@tool
def create_post():
    async def execute(submolt: str, title: str, content: str) -> str:
        """Create a new post on moltbook.

        Args:
            submolt: The submolt to post in (e.g. 'general', 'agents').
            title: Title of the post.
            content: Body content of the post.

        Returns:
            Confirmation that the post was created.
        """
        agent_posts: list = store().get("agent_posts", [])
        agent_posts.append({
            "submolt": submolt,
            "title": title,
            "content": content,
        })
        store().set("agent_posts", agent_posts)
        return f"Post created in m/{submolt}: '{title}'"

    return execute


