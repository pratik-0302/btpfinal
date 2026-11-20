import { useCallback, useEffect, useState } from "react";
import { ApiError, getHealth } from "../api";
import type { Health } from "../types";

/** Polls /api/health: fast while the model loads, slowly once it is ready. */
export function useHealth() {
  const [health, setHealth] = useState<Health | null>(null);
  const [unreachable, setUnreachable] = useState(false);

  const refresh = useCallback(async () => {
    try {
      setHealth(await getHealth());
      setUnreachable(false);
    } catch (e) {
      if (e instanceof ApiError && e.status === 0) setUnreachable(true);
    }
  }, []);

  const steady = health?.worker === "ready" || health?.worker === "busy";
  useEffect(() => {
    void refresh();
    const id = window.setInterval(() => void refresh(), steady ? 5000 : 1000);
    return () => window.clearInterval(id);
  }, [refresh, steady]);

  return { health, unreachable, refresh };
}
