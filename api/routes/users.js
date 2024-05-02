import express from "express";
import { getUser, updateUser } from "../controllers/user.js";
import { getUsers } from "../controllers/getUsers.js";

const router = express.Router();

router.get("/find/:userId", getUser);
router.put("/", updateUser);
router.get("/", getUsers); //new

router.get("/test", (req, res) => res.send("User route is working"));

export default router;
