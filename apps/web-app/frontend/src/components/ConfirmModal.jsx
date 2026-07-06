// A small confirm dialog. Usage: const [confirm, ask] = useConfirm(); ask("Delete X?", onYes)
import { createContext, useCallback, useContext, useState } from "react";

const Ctx = createContext(null);

export function ConfirmProvider({ children }) {
  const [state, setState] = useState(null); // {title, message, onConfirm}
  const ask = useCallback((title, message, onConfirm) => setState({ title, message, onConfirm }), []);
  const close = () => setState(null);
  return (
    <Ctx.Provider value={ask}>
      {children}
      {state && (
        <div className="modal-backdrop" onClick={close}>
          <div className="modal" onClick={(e) => e.stopPropagation()}>
            <h3>{state.title}</h3>
            <p>{state.message}</p>
            <div className="modal-actions">
              <button className="btn ghost" onClick={close}>Cancel</button>
              <button className="btn danger" onClick={() => { state.onConfirm(); close(); }}>Delete</button>
            </div>
          </div>
        </div>
      )}
    </Ctx.Provider>
  );
}

export const useConfirm = () => useContext(Ctx);
