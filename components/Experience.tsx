import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import type { ExperienceItem } from "@/types"

interface ExperienceProps {
  experience: ExperienceItem[]
}

export default function Experience({
  experience = [
    {
      title: "Senior Frontend Developer",
      company: "Example Corp",
      location: "Remote",
      startDate: "2021-01",
      endDate: "Present",
      description:
        "Led the development of the company's main product using React and TypeScript. Implemented new features and improved performance.",
    },
    {
      title: "Frontend Developer",
      company: "Tech Startup",
      location: "New York, NY",
      startDate: "2018-06",
      endDate: "2020-12",
      description: "Developed and maintained multiple web applications using React, Redux, and CSS-in-JS.",
    },
  ],
}: ExperienceProps) {
  return (
    <div>
      <h2 className="text-2xl font-bold mb-6">Work Experience</h2>
      <div className="space-y-4">
        {experience.map((item, index) => (
          <Card key={index}>
            <CardHeader className="pb-2">
              <CardTitle className="text-xl">{item.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="mb-2">
                <div className="font-medium">{item.company}</div>
                <div className="text-sm text-muted-foreground">{item.location}</div>
                <div className="text-sm text-muted-foreground">
                  {item.startDate} — {item.endDate}
                </div>
              </div>
              <p className="text-sm">{item.description}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )
}
