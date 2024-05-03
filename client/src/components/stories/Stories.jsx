import { useContext } from "react";
//import Stories from "../stories/Stories";
import "./stories.scss";
import { AuthContext } from "../../context/authContext";
import { useQuery } from "@tanstack/react-query";
import { makeRequest } from "../../axios";

const Stories = () => {
  const { currentUser } = useContext(AuthContext);

  const { isLoading, error, data } = useQuery(["stories"], () =>
    makeRequest.get("/stories").then((res) => {
      return res.data;
    })
  );

  //TODO Add story using react-query mutations and use upload function.

  return (
    <div className="stories">
      <div className="story">
        <img src={currentUser.profilePic} alt="" />
        <span>{currentUser.name}</span>
        <button>+</button>
      </div>
      <div className="story">
        <img src={"https://www.allprodad.com/wp-content/uploads/2021/03/05-12-21-happy-people.jpg"} alt="" />
        <span>{"DBMS"}</span>
      </div>
      <div className="story">
        <img src={"https://assets.weforum.org/article/image/XaHpf_z51huQS_JPHs-jkPhBp0dLlxFJwt-sPLpGJB0.jpg"} alt="" />
        <span>{"Rohit"}</span>
      </div>
      <div className="story">
        <img src="https://s.hdnux.com/photos/51/23/24/10827008/4/1200x0.jpg" alt="" />
        <span>{"Vishal"}</span>
      </div>
      {error
        ? ""
        : isLoading
          ? "loading"
          : data.map((story) => (
            <div className="story" key={story.id}>
              <img src={story.img} alt="" />
              <span>{story.name}</span>
            </div>
          ))}
    </div>
  );
};

export default Stories;



