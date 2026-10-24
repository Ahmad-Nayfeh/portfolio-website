export const siteConfig = {
  name: "Ahmad Nayfeh",
  title: "Ahmad Nayfeh — Electrical Design Engineer & AI Automation",
  role: "Electrical Design Engineer",
  description:
    "Electrical design engineer in medium-voltage switchgear who builds AI-assisted tools and ships complete software products.",
  url: "https://ahmadnayfeh.vercel.app",
  location: "Riyadh, Saudi Arabia",
  resume: "/resume.pdf",
  links: {
    github: "https://github.com/Ahmad-Nayfeh",
    linkedin: "https://www.linkedin.com/in/ahmad-nayfeh2000/",
  },
  contact: {
    email: "ahmadnayfeh2000@gmail.com",
    location: "Riyadh, Saudi Arabia",
  },
  nav: [
    { name: "Home", path: "/" },
    { name: "Projects", path: "/projects" },
    { name: "Notebook", path: "/blog" },
    { name: "About", path: "/about" },
  ],
} as const

/**
 * Project tiers. A project's `category` frontmatter decides its tier.
 * Anything not listed (e.g. pipeline-generated "AI Implementation" pages)
 * falls into the Lab tier.
 */
export const projectTiers = [
  {
    id: "products",
    name: "Apps",
    categories: ["Products"],
    blurb: "Complete apps I designed, built, and released, with real users.",
  },
  {
    id: "research",
    name: "Research & ML",
    categories: ["Research"],
    blurb: "Deep learning and signal processing work from my M.Sc. and independent study.",
  },
  {
    id: "lab",
    name: "Smaller projects",
    categories: [] as string[],
    blurb: "Interactive tools and experiments.",
  },
] as const

export type TierId = (typeof projectTiers)[number]["id"]

export function tierOf(category: string): TierId {
  for (const tier of projectTiers) {
    if ((tier.categories as readonly string[]).includes(category)) return tier.id
  }
  return "lab"
}
