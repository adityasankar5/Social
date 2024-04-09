import Post from "../post/Post";
import "./posts.scss";
import { useQuery } from '@tanstack/react-query';
import { makeRequest } from "../../axios";

const Posts = () => {

  const options = {
    queryKey: ['posts'], // Unique identifier for the query
    queryFn: () => makeRequest.get("/posts").then(res => res.data), // Function to fetch data
  };

  const { isLoading, error, data } = useQuery(options);

  return (
    <div className="posts">
      {isLoading ? (
        <p>Loading posts...</p>
      ) : error ? (
        <p>Error: {error.message}</p>
      ) : (
        data.map((post) => (
          <Post post={post} key={post._id} />
        ))
      )}
    </div>
  );
};

export default Posts;
