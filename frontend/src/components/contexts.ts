import { createContext } from "react";

/** Texts that explain the display-only visualisations (served by GET /api/model as new_displays). */
export const NewDisplaysContext = createContext<Record<string, string>>({});
